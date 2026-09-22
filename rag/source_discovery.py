"""Bounded official-link collection, completed before generation begins."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urljoin, urldefrag, unquote
from urllib.request import Request, build_opener, HTTPRedirectHandler
import hashlib
import json
import re
from bs4 import BeautifulSoup
from rag.budget import write_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def allowed(url, policy):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    if (parsed.scheme != 'https' or parsed.username or parsed.password or
            port not in (None, 443) or parsed.query or parsed.fragment):
        return False
    path = unquote(parsed.path)
    if '\\' in path or any(p in ('.', '..') for p in path.split('/')):
        return False
    return any(parsed.hostname == scope['host'] and any(
        path == prefix.rstrip('/') or path.startswith(prefix if prefix.endswith('/') else prefix + '/')
        for prefix in scope['paths']) for scope in policy['allow'])


class CheckedRedirect(HTTPRedirectHandler):
    def __init__(self, policy):
        self.policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed(newurl, self.policy):
            raise ValueError('Redirect outside official source boundary')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, policy, limit, timeout):
    if not allowed(url, policy):
        raise ValueError('URL outside official source boundary')
    request = Request(url, headers={'User-Agent': 'SKALA-Research-Practice/0.1'})
    with build_opener(CheckedRedirect(policy)).open(request, timeout=timeout) as response:
        if not allowed(response.url, policy):
            raise ValueError('Unexpected response URL')
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('Source exceeds byte limit')
        return raw, response.url


def links(raw, url):
    decoded = raw.decode('utf-8', errors='replace')
    soup = BeautifulSoup(decoded, 'html.parser')
    for node in soup(['nav', 'header', 'footer', 'aside']):
        node.decompose()
    body = soup.find('main') or soup
    candidates = [a.get('href') for a in body.find_all('a', href=True)]
    candidates += re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)', decoded)
    urls = [urldefrag(urljoin(url, href))[0] for href in candidates]
    return [target for target in urls if Path(urlsplit(target).path).suffix.lower()
            in ('', '.md', '.txt', '.html', '.htm', '.pdf')]


def extract(raw, url):
    if url.lower().endswith('.pdf'):
        from io import BytesIO
        from pypdf import PdfReader
        return '\n'.join(p.extract_text() or '' for p in PdfReader(BytesIO(raw)).pages)
    text = raw.decode('utf-8')
    if url.lower().endswith(('.md', '.txt')):
        return text
    soup = BeautifulSoup(text, 'html.parser')
    for node in soup(['script', 'style', 'nav', 'footer', 'header']):
        node.decompose()
    return (soup.find('main') or soup).get_text('\n', strip=True)


class Discovery:
    def __init__(self, config, root, fetcher=fetch):
        self.config, self.root, self.fetcher = config, Path(root), fetcher
        self.events, self.sources = [], {}
        self.used = 0

    def collect(self, registered, refresh=False):
        cfg = self.config
        if cfg['depth'] != 1 or not 0 < cfg['per_source'] <= 3 or not 0 < cfg['total'] <= 6:
            raise ValueError('Discovery count/depth limits exceed contract')
        if not 0 < cfg['file_bytes'] <= 5_000_000 or not 0 < cfg['aggregate_bytes'] <= 20_000_000 or not 0 < cfg['timeout'] <= 30:
            raise ValueError('Discovery transfer limits exceed contract')
        attempts = 0
        seen = {s['url'] for s in registered.values()}
        for source_id, policy in cfg['sources'].items():
            source = registered.get(source_id)
            if not source:
                continue
            if attempts >= cfg['total']:
                break
            count = 0
            try:
                if not allowed(source["url"], policy):
                    raise ValueError("Initial source outside official boundary")
                discovery_path = self.root / (source_id + '.discovery-start.raw')
                metadata_path = self.root / (source_id + '.discovery-start.json')
                # Dedicated discovery metadata owns its bytes; registered snapshots
                # may predate the bounded fetch and have a different final URL/hash.
                if discovery_path.exists() and metadata_path.exists():
                    raw_path = discovery_path
                    metadata = json.loads(metadata_path.read_text())
                else:
                    raw_path = Path(source.get('raw_path') or discovery_path)
                    metadata = source
                raw, base_url = b'', None
                if raw_path.exists():
                    raw = self._cached_raw(raw_path)
                    if metadata.get('url') != source['url']:
                        raise ValueError('Initial cache URL mismatch')
                    if not metadata.get('raw_sha256') or sha(raw) != metadata['raw_sha256']:
                        raise ValueError('Initial cache integrity mismatch')
                    base_url = metadata.get('final_url')
                    if base_url is None:
                        # Legacy caches cannot prove where relative links resolve.
                        # Refetch boundedly; never infer the final URL from the request.
                        raw = b''
                        self.events.append({'url': source['url'], 'status': 'legacy_cache_refetch',
                                            'reason': 'missing_final_url'})
                    elif not allowed(base_url, policy):
                        raise ValueError('Initial cache final URL outside official boundary')
                candidates = links(raw, base_url) if raw else []
                if refresh or not candidates:
                    raw, final = self._fetch(source['url'], policy)
                    raw_path = self.root / (source_id + '.discovery-start.raw')
                    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(raw_path).write_bytes(raw)
                    write_json(self.root / (source_id + '.discovery-start.json'),
                               {'url': source['url'], 'final_url': final, 'raw_sha256': sha(raw)})
                    self.events.append({'url': source['url'], 'status': 'start_fetched', 'raw_path': str(raw_path), 'final_url': final,
                                        'raw_sha256': sha(raw), 'accessed_at': datetime.now(timezone.utc).isoformat()})
                    candidates = links(raw, final)
                for url in candidates:
                    if count >= cfg['per_source'] or attempts >= cfg['total']:
                        break
                    if not allowed(url, policy):
                        continue
                    if url in seen:
                        self.events.append({'url': url, 'status': 'duplicate', 'discovered_from': source['url']})
                        continue
                    seen.add(url)
                    count += 1
                    attempts += 1
                    self._collect_one(url, source, policy, refresh)
            except (ValueError, OSError, UnicodeError) as exc:
                self.events.append({'url': source['url'], 'status': 'failed', 'error': type(exc).__name__})
        write_json(self.root / 'discovery_manifest.json', {'sources': self.sources, 'events': self.events,
                   'bytes_read': self.used, 'policy_sha256': sha(json.dumps(cfg, sort_keys=True).encode())})
        return self.sources

    def _cached_raw(self, path):
        size = path.stat().st_size
        if size > self.config['file_bytes'] or size > self.config['aggregate_bytes'] - self.used:
            raise ValueError('Cache exceeds byte budget')
        self.used += size
        with path.open('rb') as stream:
            raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('Cache changed while reading')
        return raw

    def _fetch(self, url, policy):
        remaining = self.config['aggregate_bytes'] - self.used
        if remaining <= 1:
            raise ValueError('Aggregate byte budget exhausted')
        # Reserve the over-limit detection byte too. A failed/partial read keeps
        # its full reservation since the transport cannot report consumed bytes.
        allowance = min(remaining, self.config['file_bytes'] + 1)
        self.used += allowance
        raw, final = self.fetcher(url, policy, allowance - 1, self.config['timeout'])
        if len(raw) > allowance - 1 or not allowed(final, policy):
            raise ValueError('Fetch boundary violation')
        self.used -= allowance - len(raw)
        return raw, final

    def _collect_one(self, url, source, policy, refresh):
        ident = 'expanded_' + sha(url.encode())[:16]
        path, raw_path = self.root / (ident + '.json'), self.root / (ident + '.raw')
        now = datetime.now(timezone.utc).isoformat()
        try:
            if path.exists() and raw_path.exists() and not refresh:
                snapshot = json.loads(path.read_text())
                raw = self._cached_raw(raw_path)
                if (snapshot['url'] != url or snapshot['raw_sha256'] != sha(raw) or
                        snapshot['text_sha256'] != sha(snapshot['text'].encode()) or
                        not allowed(snapshot['final_url'], policy)):
                    raise ValueError('Discovery cache integrity mismatch')
                snapshot['cache_status'] = 'hit'
            else:
                raw, final = self._fetch(url, policy)
                text = extract(raw, final)
                if not text.strip():
                    raise ValueError('Empty source text')
                snapshot = {'id': ident, 'url': url, 'final_url': final, 'title': url,
                    'authors': source.get('authors', 'official source'), 'text': text,
                    'raw_sha256': sha(raw), 'text_sha256': sha(text.encode()), 'accessed_at': now,
                    'discovered_from': source['url'], 'status': 'ok', 'cache_status': 'miss'}
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(raw)
                write_json(path, snapshot)
            self.sources[ident] = {k: v for k, v in snapshot.items() if k != 'text'}
            self.sources[ident].update(type='external_web', expanded=True, local_path=str(path),
                raw_path=str(raw_path), version=snapshot['text_sha256'], date='unknown')
            self.events.append({'url': url, 'status': 'ok', 'cache_status': snapshot['cache_status']})
        except (ValueError, OSError, UnicodeError) as exc:
            self.events.append({'url': url, 'status': 'failed', 'error': type(exc).__name__,
                                'discovered_from': source['url'], 'accessed_at': now})

"""Bounded HTTP worker for configured providers; stdin/stdout are private pipes.

Invoked by providers.py using an absolute script path. Keeping transport in a
child gives connection, headers, and streaming response reads one overall
subprocess deadline without platform-specific signal handlers in the caller.
"""
import json
import sys
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    try:
        config = json.load(sys.stdin)
        request = Request(config['url'], data=json.dumps(config['body'], allow_nan=False).encode(),
                          headers=config['headers'])
        with build_opener(NoRedirect()).open(request, timeout=config['timeout']) as response:
            if not 200 <= response.status < 300:
                result = {'ok': False, 'error': f'provider HTTP status {response.status}'}
            else:
                result = {'ok': True, 'data': json.load(response)}
    except HTTPError as exc:
        status = exc.code
        exc.close()
        result = {'ok': False, 'error': f'provider HTTP status {status}'}
    except (HTTPException, OSError, URLError, ValueError, TypeError, KeyError, OverflowError):
        result = {'ok': False, 'error': 'provider HTTP request failed or returned invalid JSON'}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

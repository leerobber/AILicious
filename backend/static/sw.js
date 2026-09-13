const CACHE_NAME = "ailicious-shell-v2";
const APP_SHELL = ["/", "/manifest.json", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

// Network-first, cache as offline fallback -- a deploy should always reach an already-
// installed client on its very next load. Cache-first (the previous strategy) served a
// pinned app shell forever once installed, since CACHE_NAME never changed between
// deploys: a real deploy could update the server while every existing client kept
// serving its first-ever cached index.html indefinitely, invisibly. The cache still
// gets refreshed on every successful fetch, so this keeps the offline-capable PWA
// benefit without the staleness trap.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const responseCopy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseCopy));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

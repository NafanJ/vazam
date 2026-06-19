// Same-origin fetch helper. The FastAPI app serves these pages and the API,
// and HTTP Basic auth is handled transparently by the browser.

// Optional backend override kept for parity with the old characters page
// (read from localStorage "vazam_backend"); empty string ⇒ same origin.
export function apiBase(): string {
  try {
    return (JSON.parse(localStorage.getItem("vazam_backend") || '""') || "").replace(/\/$/, "");
  } catch {
    return "";
  }
}

export async function call(path: string, opts?: RequestInit): Promise<Response> {
  // Always advertise JSON so the content-negotiated GET /characters returns the
  // data list (a browser navigation to /characters sends Accept: text/html and
  // gets the admin page instead). Caller-supplied headers win.
  const headers = { Accept: "application/json", ...(opts?.headers || {}) };
  const r = await fetch(apiBase() + path, { ...opts, headers });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`${r.status} ${t.slice(0, 140)}`.trim());
  }
  return r;
}

// The inference backend the demo talks to.
//
// This is a Cloudflare Quick Tunnel to a desktop in Michael's office. The URL is
// EPHEMERAL: `cloudflared` issues a new one every time it restarts, and the demo
// is only live while both the tunnel and `uvicorn slm.serve:app` are running.
//
// To point the site at a new tunnel: edit this line, commit, push. The Pages
// build takes ~40 s. Visitors can also override it locally via the "endpoint"
// link in the demo panel (stored in their own localStorage, never committed).
window.SLM_API = "https://ontario-adrian-guestbook-karma.trycloudflare.com";

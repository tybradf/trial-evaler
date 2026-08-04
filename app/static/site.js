document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("info-icon-btn");
  const popover = document.getElementById("info-popover");
  if (!btn || !popover) return;

  function open() {
    popover.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  }
  function close() {
    popover.hidden = true;
    btn.setAttribute("aria-expanded", "false");
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (popover.hidden) open(); else close();
  });

  document.addEventListener("click", (e) => {
    if (!popover.hidden && !popover.contains(e.target) && e.target !== btn) {
      close();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !popover.hidden) close();
  });
});

// Intro modal -- shows once per browsing session on first landing, stays
// dismissed across page navigation within that session (sessionStorage,
// not localStorage: this is a real deployed site, not a sandboxed preview,
// and "once per session, back on a fresh visit" is exactly what
// sessionStorage is for). Kept as a fully separate listener from the
// header icon-popover logic above so a missing element in one doesn't
// silently disable the other.
document.addEventListener("DOMContentLoaded", () => {
  const SESSION_KEY = "trial_eval_intro_seen";
  const backdrop = document.getElementById("intro-modal-backdrop");
  if (!backdrop) return;

  const closeBtn = document.getElementById("intro-modal-close");
  const dismissBtn = document.getElementById("intro-modal-dismiss");
  const githubLink = document.getElementById("intro-modal-github");

  function closeModal() {
    backdrop.classList.remove("visible");
    sessionStorage.setItem(SESSION_KEY, "1");
    // Wait for the fade/scale-out transition (0.22s in CSS) before hiding,
    // so the close reads as a collapse rather than an abrupt cut.
    setTimeout(() => { backdrop.hidden = true; }, 220);
  }

  if (!sessionStorage.getItem(SESSION_KEY)) {
    // Small delay so the modal reads as an intentional reveal after the
    // page has rendered, not a jarring flash the instant the DOM is ready.
    setTimeout(() => {
      backdrop.hidden = false;
      requestAnimationFrame(() => backdrop.classList.add("visible"));
    }, 400);
  }

  closeBtn?.addEventListener("click", closeModal);
  dismissBtn?.addEventListener("click", closeModal);
  githubLink?.addEventListener("click", closeModal);

  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeModal(); // backdrop itself, not the card inside it
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !backdrop.hidden) closeModal();
  });
});

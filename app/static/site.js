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

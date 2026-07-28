document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll("#case-picker button");
  const cards = document.querySelectorAll(".case-card");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const index = btn.dataset.index;
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      cards.forEach((card) => {
        card.style.display = card.dataset.caseIndex === index ? "" : "none";
      });
    });
  });
});

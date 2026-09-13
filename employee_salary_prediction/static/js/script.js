document.addEventListener("DOMContentLoaded", function () {
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.querySelector(".navbar-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
  }

  const deleteForms = document.querySelectorAll("[data-confirm-delete]");
  deleteForms.forEach(function (form) {
    form.addEventListener("submit", function (event) {
      const confirmed = window.confirm("Delete this prediction record? This cannot be undone.");
      if (!confirmed) {
        event.preventDefault();
      }
    });
  });

  const registrationForm = document.querySelector("#registration-form");
  if (registrationForm) {
    registrationForm.addEventListener("submit", function (event) {
      const password = registrationForm.querySelector("#password").value;
      const confirmPassword = registrationForm.querySelector("#confirm_password").value;
      if (password !== confirmPassword) {
        event.preventDefault();
        window.alert("Passwords do not match.");
      }
    });
  }

  const skillLimit = 8;
  const skillCheckboxes = document.querySelectorAll(".skill-chip input[type=checkbox]");
  if (skillCheckboxes.length > 0) {
    skillCheckboxes.forEach(function (checkbox) {
      checkbox.addEventListener("change", function () {
        const checkedCount = document.querySelectorAll(".skill-chip input[type=checkbox]:checked").length;
        if (checkedCount > skillLimit) {
          checkbox.checked = false;
          window.alert("Please select up to " + skillLimit + " skills.");
        }
      });
    });
  }

  const predictionForm = document.querySelector("#prediction-form");
  if (predictionForm) {
    predictionForm.addEventListener("submit", function (event) {
      const experienceInput = predictionForm.querySelector("#years_experience");
      const ageInput = predictionForm.querySelector("#age");
      if (experienceInput && ageInput) {
        const experience = parseFloat(experienceInput.value || "0");
        const age = parseFloat(ageInput.value || "0");
        if (experience > age - 16) {
          event.preventDefault();
          window.alert("Years of experience seems too high for the age entered. Please check your inputs.");
        }
      }
    });
  }
});

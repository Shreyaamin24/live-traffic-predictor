let interval = null;

document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("startBtn");
  const outputBox = document.getElementById("output");
  const inputList = document.getElementById("inputValues");
  const predictionText = document.getElementById("prediction");

  startBtn.addEventListener("click", () => {
    outputBox.style.display = "block";

    if (interval) clearInterval(interval);

    interval = setInterval(async () => {
      try {
        const response = await fetch("/predict");
        const data = await response.json();

        inputList.innerHTML = "";
        for (let key in data.input) {
          const li = document.createElement("li");
          li.textContent = `${key}: ${data.input[key]}`;
          inputList.appendChild(li);
        }

        const pred = data.prediction.trim();
        predictionText.textContent = pred;

        // Remove previous color classes
        predictionText.classList.remove("text-danger", "text-warning", "text-success");

        // Add color based on prediction
        if (pred === "High") {
          predictionText.classList.add("text-danger");
        } else if (pred === "Medium") {
          predictionText.classList.add("text-warning");
        } else if (pred === "Low") {
          predictionText.classList.add("text-success");
        }

      } catch (err) {
        console.error("Error fetching prediction:", err);
      }
    }, 5000);
  });
});

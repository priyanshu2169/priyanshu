let model;

// List of readable class names
const classNames = [
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
  "Cherry_(including_sour)_Powdery_mildew",
  "Cherry_(including_sour)_healthy",
  "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)Common_rust",
  "Corn_(maize)_Northern_Leaf_Blight",
  "Corn_(maize)_healthy",
  "Grape___Black_rot",
  "Grape__Esca(Black_Measles)",
  "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
  "Grape___healthy",
  "Orange__Haunglongbing(Citrus_greening)",
  "Peach___Bacterial_spot",
  "Peach___healthy",
  "Pepper,bell__Bacterial_spot",
  "Pepper,bell__healthy",
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  "Raspberry___healthy",
  "Soybean___healthy",
  "Squash___Powdery_mildew",
  "Strawberry___Leaf_scorch",
  "Strawberry___healthy",
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite",
  "Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___healthy"
];

// Load model
async function loadModel() {
  model = await tf.loadLayersModel("model_web/model.json");
  console.log("✅ Model loaded");
}
loadModel();

// Format label nicely
function formatLabel(label) {
  return label.replace(/_/g, " ").replace(/\./g, " ");
}

// Run prediction on image
async function runImagePrediction() {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  const imgElement = document.getElementById("preview");

  // Preprocess the image
  const tensor = tf.browser.fromPixels(imgElement)
    .resizeNearestNeighbor([128, 128])  // match model input size
    .toFloat()
    .div(255.0)
    .expandDims();

  // Predict
  const result = await model.predict(tensor).data();

  // Get top 3 predictions
  const topPredictions = Array.from(result)
    .map((prob, index) => ({
      label: classNames[index],
      confidence: prob
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  // Display result
  const outputEl = document.getElementById("output");
  outputEl.innerHTML = `
    <strong>Top Predictions:</strong><br>
    <ul>
      ${topPredictions
        .map(pred => `<li><strong>${formatLabel(pred.label)}</strong>: ${(pred.confidence * 100).toFixed(2)}%</li>`)
        .join("")}
    </ul>
  `;

  // Cleanup tensors
  tensor.dispose();
}
  
// Image preview handler
document.getElementById("imageInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("preview");
    img.src = e.target.result;
    img.style.display = "block";
  };
  reader.readAsDataURL(file);
});

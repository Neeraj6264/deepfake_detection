// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [file, setFile] = useState(null);
//   const [result, setResult] = useState(null);

//   const uploadImage = async () => {
//     const formData = new FormData();
//     formData.append("image", file);

//     const res = await axios.post("http://localhost:3000/detect", formData);
//     setResult(res.data);
//   };

//   return (
//     <div style={{ textAlign: "center" }}>
//       <h1>Deepfake Detector 🔍</h1>

//       <input type="file" onChange={(e) => setFile(e.target.files[0])} />
//       <br /><br />

//       <button onClick={uploadImage}>Detect</button>

//       {result && (
//         <div>
//           <h2>{result.label}</h2>
//           <p>Confidence: {result.confidence}%</p>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
 import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const uploadImage = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:3000/detect", formData);
      setResult(res.data);
    } catch (err) {
      console.log(err);
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      
      {/* HERO SECTION */}
      <h1 style={styles.title}>Deepfake Image Detection Online</h1>
      <p style={styles.subtitle}>
        Detect AI-generated faces with real-time analysis and confidence score.
      </p>

      {/* MAIN GRID */}
      <div style={styles.grid}>

        {/* LEFT SIDE (UPLOAD BOX) */}
        <div style={styles.uploadBox}>
          <input type="file" onChange={handleFile} />

          <p style={{ marginTop: 10 }}>
            Drag & drop image or click
          </p>

          <button onClick={uploadImage} style={styles.button}>
            {loading ? "Detecting..." : "Detect Forgery"}
          </button>
        </div>

        {/* RIGHT SIDE (PREVIEW + RESULT) */}
        <div style={styles.previewBox}>
          
          {preview ? (
            <img src={preview} alt="preview" style={styles.image} />
          ) : (
            <p>No image selected</p>
          )}

          {result && (
            <div style={styles.result}>
              <h2>{result.label}</h2>
              <p>Confidence: {result.confidence}%</p>

              <div
                style={{
                  ...styles.badge,
                  backgroundColor:
                    result.label === "FAKE" ? "#ef4444" : "#22c55e",
                }}
              >
                {result.label === "FAKE"
                  ? "AI Generated"
                  : "Real Human"}
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#f8fafc",
    textAlign: "center",
    padding: "40px",
    fontFamily: "Arial",
  },
  title: {
    fontSize: "40px",
    fontWeight: "bold",
  },
  subtitle: {
    color: "#555",
    marginBottom: "30px",
  },
  grid: {
    display: "flex",
    justifyContent: "center",
    gap: "40px",
    flexWrap: "wrap",
  },
  uploadBox: {
    border: "2px dashed #ccc",
    padding: "30px",
    borderRadius: "12px",
    width: "300px",
    background: "#fff",
  },
  previewBox: {
    width: "300px",
    background: "#fff",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 5px 15px rgba(0,0,0,0.1)",
  },
  image: {
    width: "100%",
    borderRadius: "10px",
  },
  button: {
    marginTop: "15px",
    padding: "10px 20px",
    borderRadius: "8px",
    border: "none",
    background: "#2563eb",
    color: "white",
    cursor: "pointer",
  },
  result: {
    marginTop: "15px",
  },
  badge: {
    marginTop: "10px",
    padding: "6px 12px",
    borderRadius: "20px",
    color: "white",
    display: "inline-block",
  },
};

export default App;
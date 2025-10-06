const express = require("express");
const fs = require("fs");
const path = require("path");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const cors = require("cors");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");

const app = express();
app.use(express.json());
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(__dirname));

const USERS_FILE = path.join(__dirname, "users.json");
const SECRET_KEY = "your_secret_key";

// ---------------- Helper Functions ----------------
function getUsers() {
  if (!fs.existsSync(USERS_FILE)) return [];
  return JSON.parse(fs.readFileSync(USERS_FILE, "utf8"));
}
function saveUsers(users) {
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}

// ---------------- REGISTER ----------------
app.post("/register", async (req, res) => {
  const { username, email, password } = req.body;
  if (!username || !email || !password) return res.status(400).json({ message: "All fields required" });

  const users = getUsers();
  const normalizedEmail = email.trim().toLowerCase();

  if (users.find(u => u.email === normalizedEmail)) return res.status(400).json({ message: "User exists" });

  const hashedPassword = await bcrypt.hash(password, 10);
  users.push({ username, email: normalizedEmail, password: hashedPassword, profile: {} });
  saveUsers(users);
  res.json({ message: "Registration successful" });
});

// ---------------- LOGIN ----------------
app.post("/login", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) return res.status(400).json({ message: "All fields required" });

  const users = getUsers();
  const normalizedEmail = email.trim().toLowerCase();
  const user = users.find(u => u.email === normalizedEmail);
  if (!user) return res.status(404).json({ message: "User not found" });

  const isMatch = await bcrypt.compare(password, user.password);
  if (!isMatch) return res.status(401).json({ message: "Invalid password" });

  const token = jwt.sign({ email: user.email }, SECRET_KEY, { expiresIn: "1h" });
  res.json({ message: "Login successful", token, email: user.email });
});

// ---------------- UPDATE PROFILE ----------------
app.post("/profile", (req, res) => {
  const { email, username, degree, specialization, cgpa, graduationYear, skills } = req.body;
  if (!email) return res.status(400).json({ message: "Email required" });

  const users = getUsers();
  const idx = users.findIndex(u => u.email === email);
  if (idx === -1) return res.status(404).json({ message: "User not found" });

  users[idx].profile = {
    username,
    degree,
    specialization,
    cgpa: parseFloat(cgpa) || 0,
    graduationYear: parseInt(graduationYear) || null,
    skills: skills ? skills.split(",").map(s => s.trim()) : [],
    predictions: users[idx].profile.predictions || []
  };

  saveUsers(users);
  res.json({ message: "Profile updated", profile: users[idx].profile });
});

// ---------------- GET PROFILE ----------------
app.get("/profile", (req, res) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  let email;
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    email = decoded.email;
  } catch {
    return res.status(401).json({ error: "Invalid token" });
  }

  const users = getUsers();
  const user = users.find((u) => u.email === email);
  if (!user) return res.status(404).json({ message: "User not found" });

  res.json({
    username: user.username,
    email: user.email,
    ...user.profile
  });
});

// ---------------- PREDICT ----------------
// app.post("/predict", (req, res) => {
//   const token = req.headers.authorization?.split(" ")[1];
//   if (!token) return res.status(401).json({ error: "Unauthorized" });

//   let email;
//   try {
//     const decoded = jwt.verify(token, SECRET_KEY);
//     email = decoded.email;
//   } catch {
//     return res.status(401).json({ error: "Invalid token" });
//   }

//   const inputData = req.body;
//   const scriptPath = path.join(__dirname, "ml", "predict.py");

//   const python = spawn("python", [scriptPath, JSON.stringify(inputData)]);
//   let result = "";

//   python.stdout.on("data", (data) => { result += data.toString(); });
//   python.stderr.on("data", (data) => { console.error("Python error:", data.toString()); });

//   python.on("close", () => {
//     if (!result) return res.status(500).json({ message: "Prediction failed" });

//     const predictedRole = result.trim();
//     const users = getUsers();
//     const idx = users.findIndex(u => u.email === email);
//     if (idx !== -1) {
//       if (!users[idx].profile) users[idx].profile = {};
//       if (!Array.isArray(users[idx].profile.predictions)) users[idx].profile.predictions = [];

//       users[idx].profile.predictions.push({
//         cgpa: inputData.CGPA || "",
//         degree: inputData.Degree || "",
//         major: inputData.Major || "",
//         skills: Array.isArray(inputData.Skills) ? inputData.Skills : [],
//         certifications: Array.isArray(inputData.Certifications) ? inputData.Certifications : [],
//         experience: inputData.Experience || "",
//         employed: inputData.Employed || "",
//         industryPreference: inputData.IndustryPreference || "",
//         predictedRole,
//         date: new Date().toISOString()
//       });

//       saveUsers(users);
//       console.log("Saved prediction:", users[idx].profile.predictions.slice(-1)[0]);
//     }

//     res.json({ predictedRole });
//   });
// });
// ---------------- PREDICT ----------------
app.post("/predict", (req, res) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  let email;
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    email = decoded.email;
  } catch {
    return res.status(401).json({ error: "Invalid token" });
  }

  const inputData = req.body;
  const scriptPath = path.join(__dirname, "ml", "predict.py");

  const python = spawn("python", [scriptPath, JSON.stringify(inputData)]);
  let result = "";

  python.stdout.on("data", (data) => { result += data.toString(); });
  python.stderr.on("data", (data) => { console.error("Python error:", data.toString()); });

  python.on("close", () => {
    if (!result) return res.status(500).json({ message: "Prediction failed" });

    let predictions;
    try {
      predictions = JSON.parse(result.trim()); // [{role, confidence}, ...]
    } catch (err) {
      console.error("Parse error:", err, result);
      return res.status(500).json({ message: "Prediction parse failed" });
    }

    const users = getUsers();
    const idx = users.findIndex(u => u.email === email);
    if (idx !== -1) {
      if (!users[idx].profile) users[idx].profile = {};
      if (!Array.isArray(users[idx].profile.predictions)) users[idx].profile.predictions = [];

      users[idx].profile.predictions.push({
        ...inputData,
        predictions,
        date: new Date().toISOString()
      });

      saveUsers(users);
      console.log("Saved prediction:", users[idx].profile.predictions.slice(-1)[0]);
    }

    // res.json({ predictions });
    res.json({ 
  predictedRole: predictions[0]?.role || null, 
  predictions 
});

  });
});

// ---------------- HISTORY ----------------
app.get("/history", (req, res) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  let email;
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    email = decoded.email;
  } catch {
    return res.status(401).json({ error: "Invalid token" });
  }

  const users = getUsers();
  const user = users.find(u => u.email === email);
  res.json(user?.profile?.predictions || []);
});

// ---------------- START SERVER ----------------
app.listen(5000, () => console.log("🚀 Server running on http://localhost:5000"));

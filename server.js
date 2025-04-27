const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Dummy AI prediction endpoint
app.post('/predict', (req, res) => {
    const data = req.body;
    console.log(data);

    // Dummy logic: You will replace this with real AI model prediction
    const prediction = data.age > 50 ? 'High Risk' : 'Low Risk';

    res.json({ prediction: prediction });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});

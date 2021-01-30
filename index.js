import express from "express"
import fs from "fs"
import multer from "multer"
import * as tf from "@tensorflow/tfjs-node"

const upload = multer({ dest: 'uploads/' })

const app = express()
const model = await tf.loadLayersModel("file://D:/hotdog_no_hotdog/hotdog_model/model.json")
model.compile({ loss: "binaryCrossentropy", optimizer: "adam" })
const IMAGE_HEIGHT = 128
const IMAGE_WIDTH = 128
app.use(express.static("static"))

app.post("/predict", upload.single('image'), (req,res) => {
    if(req.file) {
        console.log("file received...")
        let image = fs.readFileSync(req.file.path)
        let decoded_img = tf.node.decodeImage(image)
        let img = tf.image.resizeBilinear(decoded_img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = img.div(255.0)
        img = img.reshape([1,IMAGE_HEIGHT,IMAGE_WIDTH,3])
        console.log("Making prediction...")
        let prediction = model.predict(img)
        if (prediction.dataSync()[0] >= 0.5){
            res.send("HOTDOG")
        }
        if (prediction.dataSync()[0] < 0.5){
            res.send("NOT HOTDOG")
        }
        let files = fs.readdirSync("uploads")
        files.forEach(file => {
            fs.unlinkSync(`uploads/${file}`)
        })
    }
})

app.listen(3200, () => console.log("listening on port 3200"))
use anyhow::Ok;
use executorch_yolox::yolox;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::image::{self, ImageBuffer, Rgb};
use imageproc::rect::Rect;
use std::vec;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};

fn draw_rect(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x1: f32, y1: f32, x2: f32, y2: f32) {
    let x1 = x1 as u32;
    let y1 = y1 as u32;
    let x2 = x2 as u32;
    let y2 = y2 as u32;
    let rect = Rect::at(x1 as i32, y1 as i32).of_size(x2 - x1 as u32, (y2 - y1) as u32);
    draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
}

fn get_coco_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resource/models")
        .join("coco-classes.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}

fn main() -> anyhow::Result<()> {
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = base_path.join("resource/models/yolox_tiny_coreml.pte");

    let mut yolox = yolox::YoloX::new(&model_path, vec![1, 3, 416, 416], get_coco_labels());
    yolox.load()?;

    let image_dir = base_path.join("resource/images/input");
    let image_dir = fs::read_dir(image_dir)?;

    let mut all_images: Vec<String> = image_dir
        .map(|entry| entry.unwrap().path().display().to_string())
        .collect();
    all_images.sort();
    for (i, image_path) in all_images.iter().enumerate() {
        let image_buffer = image::open(image_path)?;
        let image_buffer = image_buffer.to_rgb8();
        let (resized_image, mut resized_image_buffer) = yolox.pre_processing(&image_buffer);
        let tensor = yolox.forward(&resized_image)?;
        let preds = yolox.post_processing(&tensor);

        for pred in preds {
            let (class, score, x1, y1, x2, y2) = pred;
            println!(
                "class: {}, score: {}, x1: {}, y1: {}, x2: {}, y2: {}",
                class, score, x1, y1, x2, y2
            );

            draw_rect(&mut resized_image_buffer, x1, y1, x2, y2);
        }

        resized_image_buffer.save(format!("resource/images/output/predict_{}.jpg", i))?;
    }

    Ok(())
}

use berry_executorch::{module::Module, ExecutorchError, Tensor};
use imageproc::image::{
    imageops::{self, FilterType},
    ImageBuffer, Pixel, Rgb,
};
use std::path::PathBuf;

#[derive(Debug)]
pub struct YoloX {
    module: Module,
    input_sizes: Vec<usize>,
    classes: Vec<String>,
}

impl YoloX {
    pub fn new(model_path: &PathBuf, input_sizes: Vec<usize>, classes: Vec<String>) -> Self {
        let module = Module::new(&model_path.display().to_string()).unwrap();
        Self {
            module,
            input_sizes,
            classes,
        }
    }

    pub fn load(&mut self) -> Result<(), ExecutorchError> {
        self.module.load()
    }

    pub fn forward(&mut self, image: &[f32]) -> Result<Tensor, ExecutorchError> {
        let input_sizes = self
            .input_sizes
            .iter()
            .map(|x| *x as i32)
            .collect::<Vec<_>>();
        self.module.forward(image, &input_sizes)
    }

    pub fn pre_processing(
        &self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> (Vec<f32>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
        let (h, w) = (self.input_sizes[2], self.input_sizes[3]);
        let image_buffer = self.padding_image(image);
        let image_buffer = imageops::resize(&image_buffer, w as u32, h as u32, FilterType::Nearest);

        // convert image to Vec<f32> with channel first format
        let mut image = vec![0.0; 3 * h as usize * w as usize];
        for j in 0..h {
            for i in 0..w {
                let pixel = image_buffer.get_pixel(i as u32, j as u32);
                let channels = pixel.channels();
                for c in 0..3 {
                    image[c * h * w + j * w + i] = channels[c] as f32;
                }
            }
        }
        (image, image_buffer)
    }

    fn padding_image(
        &self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = image.dimensions();
        let target_size = if width > height { width } else { height };
        let mut new_image = ImageBuffer::new(target_size as u32, target_size as u32);
        let x_offset = (target_size as u32 - width) / 2;
        let y_offset = (target_size as u32 - height) / 2;
        for j in 0..height {
            for i in 0..width {
                let pixel = image.get_pixel(i, j);
                new_image.put_pixel(i + x_offset, j + y_offset, *pixel);
            }
        }
        new_image
    }

    pub fn post_processing(&self, preds: &Tensor) -> Vec<(String, f32, f32, f32, f32, f32)> {
        let preds = &preds.data;
        let mut positions = vec![];
        let mut classes = vec![];
        let mut objectnesses = vec![];
        for i in 0..3549 {
            let offset = i * 85;
            let objectness = preds[offset + 4];

            let (class, score) = preds[offset + 5..offset + 85]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            let class = self.classes[class].clone();
            let x1 = preds[offset];
            let y1 = preds[offset + 1];
            let x2 = preds[offset + 2];
            let y2 = preds[offset + 3];
            classes.push((class, score));
            positions.push((x1, y1, x2, y2));
            objectnesses.push(objectness);
        }

        let locs = self.calc_loc(&positions, &self.input_sizes);

        let mut result = vec![];
        // filter by objectness
        let indices = self.non_max_suppression(&locs, &objectnesses, 0.9, 0.2);
        for bbox in indices {
            let (i, (x, y, w, h)) = bbox;
            let (class, &score) = &classes[i];
            result.push((class.clone(), score, x, y, w, h));
        }
        result
    }

    fn calc_loc(
        &self,
        positions: &Vec<(f32, f32, f32, f32)>,
        input_size: &Vec<usize>,
    ) -> Vec<(f32, f32, f32, f32)> {
        let mut locs = vec![];

        // calc girds
        let (h, w) = (input_size[2], input_size[3]);
        let strides = vec![8, 16, 32];
        let mut h_grids = vec![];
        let mut w_grids = vec![];

        for stride in strides.iter() {
            let mut h_grid = vec![0.0; h / stride];
            let mut w_grid = vec![0.0; w / stride];

            for i in 0..h / stride {
                h_grid[i] = i as f32;
            }
            for i in 0..w / stride {
                w_grid[i] = i as f32;
            }
            h_grids.push(h_grid);
            w_grids.push(w_grid);
        }
        let acc = vec![0, 52 * 52, 52 * 52 + 26 * 26, 52 * 52 + 26 * 26 + 13 * 13];

        for (i, stride) in strides.iter().enumerate() {
            let h_grid = &h_grids[i];
            let w_grid = &w_grids[i];
            let idx = acc[i];

            for (i, y) in h_grid.iter().enumerate() {
                for (j, x) in w_grid.iter().enumerate() {
                    let p = idx + i * w / stride + j;
                    let (px, py, pw, ph) = positions[p];
                    let (x, y) = ((x + px) * *stride as f32, (y + py) * *stride as f32);
                    let (ww, hh) = (pw.exp() * *stride as f32, ph.exp() * *stride as f32);
                    let loc = (x - ww / 2.0, y - hh / 2.0, x + ww / 2.0, y + hh / 2.0);
                    locs.push(loc);
                }
            }
        }
        locs
    }

    fn non_max_suppression(
        &self,
        boxes: &Vec<(f32, f32, f32, f32)>,
        scores: &Vec<f32>,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<(usize, (f32, f32, f32, f32))> {
        let mut new_boxes = vec![];
        let mut sorted_indices = (0..boxes.len()).collect::<Vec<_>>();
        sorted_indices.sort_by(|a, b| scores[*a].partial_cmp(&scores[*b]).unwrap());

        while let Some(last) = sorted_indices.pop() {
            let mut remove_list = vec![];
            let score = scores[last];
            let bbox = boxes[last];
            let mut numerator = (
                bbox.0 * score,
                bbox.1 * score,
                bbox.2 * score,
                bbox.3 * score,
            );
            let mut denominator = score;

            for i in 0..sorted_indices.len() {
                let idx = sorted_indices[i];
                let (x1, y1, x2, y2) = boxes[idx];
                let (x1_, y1_, x2_, y2_) = boxes[last];
                let box1_area = (x2 - x1) * (y2 - y1);

                let inter_x1 = x1.max(x1_);
                let inter_y1 = y1.max(y1_);
                let inter_x2 = x2.min(x2_);
                let inter_y2 = y2.min(y2_);
                let inter_w = (inter_x2 - inter_x1).max(0.0);
                let inter_h = (inter_y2 - inter_y1).max(0.0);
                let inter_area = inter_w * inter_h;
                let area1 = (x2 - x1) * (y2 - y1);
                let area2 = (x2_ - x1_) * (y2_ - y1_);
                let union_area = area1 + area2 - inter_area;
                let iou = inter_area / union_area;

                if scores[idx] < score_threshold {
                    remove_list.push(i);
                } else if iou > iou_threshold {
                    remove_list.push(i);
                    let w = scores[idx] * iou;
                    numerator = (
                        numerator.0 + boxes[idx].0 * w,
                        numerator.1 + boxes[idx].1 * w,
                        numerator.2 + boxes[idx].2 * w,
                        numerator.3 + boxes[idx].3 * w,
                    );
                    denominator += w;
                } else if inter_area / box1_area > 0.7 {
                    remove_list.push(i);
                }
            }
            for i in remove_list.iter().rev() {
                sorted_indices.remove(*i);
            }
            let new_bbox = (
                numerator.0 / denominator,
                numerator.1 / denominator,
                numerator.2 / denominator,
                numerator.3 / denominator,
            );
            new_boxes.push((last, new_bbox));
        }
        new_boxes
    }
}

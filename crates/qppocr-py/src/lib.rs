//! Python bindings for the `qppocr` pure-Rust PP-OCRv6 engine (PyO3).
//!
//! Build/install with maturin (see the repo-root `pyproject.toml`):
//!
//! ```bash
//! maturin build --release            # wheel in target/wheels
//! maturin develop --release          # install into the active interpreter
//! ```
//!
//! ```python
//! import qppocr
//! eng = qppocr.Engine(tier="small", models="models")
//! r = eng.run("receipt.png")          # path (str/PathLike) or bytes
//! for line in r.lines:
//!     print(line.confidence, line.text, line.pts)
//! ```

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use ::qppocr as qp;

/// Map an engine error onto a Python exception (bad input -> ValueError).
fn map_err(e: qp::Error) -> PyErr {
    match e {
        qp::Error::Model(_) | qp::Error::Image(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn pts_of(p: [[f32; 2]; 4]) -> Vec<(f32, f32)> {
    p.iter().map(|q| (q[0], q[1])).collect()
}

fn box_of(p: [[f32; 2]; 4]) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    (p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1])
}

fn parse_tier(s: &str) -> PyResult<qp::Tier> {
    match s {
        "tiny" => Ok(qp::Tier::Tiny),
        "small" => Ok(qp::Tier::Small),
        "medium" => Ok(qp::Tier::Medium),
        _ => Err(PyValueError::new_err("tier: tiny | small | medium")),
    }
}

fn parse_preset(s: &str) -> PyResult<qp::Preset> {
    match s {
        "speed" => Ok(qp::Preset::Speed),
        "balanced" => Ok(qp::Preset::Balanced),
        "accuracy" => Ok(qp::Preset::Accuracy),
        _ => Err(PyValueError::new_err("preset: speed | balanced | accuracy")),
    }
}

/// One character's quad in original-image coordinates.
#[pyclass(name = "Char", skip_from_py_object)]
#[derive(Clone)]
pub struct Char {
    /// The character (a single `char`; pixel-detected spaces included).
    #[pyo3(get)]
    pub text: String,
    pts: [[f32; 2]; 4],
}

#[pymethods]
impl Char {
    /// Quad corners TL, TR, BR, BL as `[(x, y); 4]`, original-image pixels.
    #[getter]
    fn pts(&self) -> Vec<(f32, f32)> {
        pts_of(self.pts)
    }

    /// Same quad flattened to `(x0, y0, x1, y1, x2, y2, x3, y3)`.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
        box_of(self.pts)
    }

    fn __repr__(&self) -> String {
        format!("Char({:?})", self.text)
    }
}

/// A recognized text line.
#[pyclass(name = "Line", skip_from_py_object)]
#[derive(Clone)]
pub struct Line {
    /// Decoded text (may be empty = detected but unreadable).
    #[pyo3(get)]
    pub text: String,
    /// CTC confidence.
    #[pyo3(get)]
    pub confidence: f32,
    /// 0 or 180 (180 = classifier flipped an upside-down line).
    #[pyo3(get)]
    pub rotation: i32,
    /// True if this line came from the second (region-retry) detection pass.
    #[pyo3(get)]
    pub retried: bool,
    chars: Vec<Char>,
    pts: [[f32; 2]; 4],
}

#[pymethods]
impl Line {
    /// Per-character quads, aligned with `text`.
    #[getter]
    fn chars(&self) -> Vec<Char> {
        self.chars.clone()
    }

    /// Line quad corners TL, TR, BR, BL as `[(x, y); 4]`, original-image pixels.
    #[getter]
    fn pts(&self) -> Vec<(f32, f32)> {
        pts_of(self.pts)
    }

    /// Same quad flattened to `(x0, y0, x1, y1, x2, y2, x3, y3)`.
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
        box_of(self.pts)
    }

    fn __repr__(&self) -> String {
        format!("Line({:?}, confidence={:.3})", self.text, self.confidence)
    }
}

/// Per-stage millisecond timings (fields match the CLI's JSON keys).
#[pyclass(name = "Timings", skip_from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct Timings {
    /// det preprocessing.
    #[pyo3(get)]
    pub det_pre_ms: f64,
    /// det forward pass.
    #[pyo3(get)]
    pub det_infer_ms: f64,
    /// det postprocessing.
    #[pyo3(get)]
    pub det_post_ms: f64,
    /// Crop.
    #[pyo3(get)]
    pub crop_ms: f64,
    /// Direction classification.
    #[pyo3(get)]
    pub cls_ms: f64,
    /// rec preprocessing.
    #[pyo3(get)]
    pub rec_pre_ms: f64,
    /// rec forward pass.
    #[pyo3(get)]
    pub rec_infer_ms: f64,
    /// rec postprocessing.
    #[pyo3(get)]
    pub rec_post_ms: f64,
    /// Total engine time for the image.
    #[pyo3(get)]
    pub total_ms: f64,
}

/// One `run()` result.
#[pyclass(name = "Result")]
pub struct OcrOut {
    /// Text lines in reading order.
    #[pyo3(get)]
    pub lines: Vec<Line>,
    /// Source image width (after EXIF orientation).
    #[pyo3(get)]
    pub width: i32,
    /// Source image height.
    #[pyo3(get)]
    pub height: i32,
    /// Detector working-copy width (after the long-side cap).
    #[pyo3(get)]
    pub work_w: i32,
    /// Detector working-copy height.
    #[pyo3(get)]
    pub work_h: i32,
    /// Detector network input width.
    #[pyo3(get)]
    pub det_input_w: i32,
    /// Detector network input height.
    #[pyo3(get)]
    pub det_input_h: i32,
    /// Raw detection-box count (before merging).
    #[pyo3(get)]
    pub num_boxes: usize,
    /// Lines merged.
    #[pyo3(get)]
    pub num_merged: usize,
    /// Clutter boxes tightened.
    #[pyo3(get)]
    pub num_decluttered: usize,
    /// Region-retry passes.
    #[pyo3(get)]
    pub num_det_retried: usize,
    /// Lines flipped by the classifier.
    #[pyo3(get)]
    pub num_flipped: usize,
    /// Detected-but-unreadable lines.
    #[pyo3(get)]
    pub num_unread: usize,
    /// Per-stage timings.
    #[pyo3(get)]
    pub timings: Timings,
}

fn from_core(r: qp::OcrResult, w: i32, h: i32) -> OcrOut {
    let lines = r
        .lines
        .into_iter()
        .map(|l| Line {
            text: l.text,
            confidence: l.confidence,
            rotation: l.rotation,
            retried: l.retried,
            chars: l
                .chars
                .into_iter()
                .map(|c| Char {
                    text: c.text,
                    pts: c.pts,
                })
                .collect(),
            pts: l.pts,
        })
        .collect();
    let t = r.timings;
    OcrOut {
        lines,
        width: w,
        height: h,
        work_w: r.work_w,
        work_h: r.work_h,
        det_input_w: r.det_input_w,
        det_input_h: r.det_input_h,
        num_boxes: r.num_boxes,
        num_merged: r.num_merged,
        num_decluttered: r.num_decluttered,
        num_det_retried: r.num_det_retried,
        num_flipped: r.num_flipped,
        num_unread: r.num_unread,
        timings: Timings {
            det_pre_ms: t.det_pre_ms,
            det_infer_ms: t.det_infer_ms,
            det_post_ms: t.det_post_ms,
            crop_ms: t.crop_ms,
            cls_ms: t.cls_ms,
            rec_pre_ms: t.rec_pre_ms,
            rec_infer_ms: t.rec_infer_ms,
            rec_post_ms: t.rec_post_ms,
            total_ms: t.total_ms,
        },
    }
}

#[pymethods]
impl OcrOut {
    fn __repr__(&self) -> String {
        format!(
            "Result({} lines, {}x{}, {:.1} ms)",
            self.lines.len(),
            self.width,
            self.height,
            self.timings.total_ms
        )
    }
}

/// The OCR engine: build once, call `run()` many times (thread-safe).
#[pyclass(name = "Engine")]
pub struct Engine {
    inner: qp::Engine,
}

enum Source {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

fn extract_source(obj: &Bound<'_, PyAny>) -> PyResult<Source> {
    if let Ok(b) = obj.extract::<Vec<u8>>() {
        return Ok(Source::Bytes(b));
    }
    if let Ok(p) = obj.extract::<PathBuf>() {
        return Ok(Source::Path(p));
    }
    Err(PyTypeError::new_err(
        "image must be a path (str / os.PathLike) or bytes",
    ))
}

#[pymethods]
impl Engine {
    /// Build an engine.
    ///
    /// Args:
    ///     tier: `"tiny" | "small" | "medium"` (default `"small"`).
    ///     models: model directory (default `"models"`).
    ///     preset: `"speed" | "balanced" | "accuracy"` (default: engine default).
    ///     threads: worker threads; 0 = auto.
    ///     detect_orientation: run the 0/180 classifier (if `cls.onnx` exists).
    ///     verify_sha256: verify det/rec weights against the upstream SHA-256.
    #[new]
    #[pyo3(signature = (tier = "small", models = "models", preset = None, threads = 0, detect_orientation = true, verify_sha256 = true))]
    fn new(
        tier: &str,
        models: &str,
        preset: Option<&str>,
        threads: usize,
        detect_orientation: bool,
        verify_sha256: bool,
    ) -> PyResult<Self> {
        let mut b = qp::Engine::builder()
            .tier(parse_tier(tier)?)
            .detect_orientation(detect_orientation)
            .verify_sha256(verify_sha256);
        if let Some(p) = preset {
            b = b.preset(parse_preset(p)?);
        }
        if threads > 0 {
            b = b.threads(threads);
        }
        b.build(models).map(|inner| Engine { inner }).map_err(map_err)
    }

    /// Full pipeline (det -> cls -> rec); `image` is a path or bytes.
    fn run(&self, py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<OcrOut> {
        match extract_source(image)? {
            Source::Path(p) => {
                let (r, w, h) = py
                    .detach(|| -> Result<_, qp::Error> {
                        let img = qp::decode_file(&p)?;
                        let (w, h) = (img.w, img.h);
                        Ok((self.inner.run(&img)?, w, h))
                    })
                    .map_err(map_err)?;
                Ok(from_core(r, w, h))
            }
            Source::Bytes(b) => {
                let (r, w, h) = py
                    .detach(|| -> Result<_, qp::Error> {
                        let img = qp::decode_bytes(&b)?;
                        let (w, h) = (img.w, img.h);
                        Ok((self.inner.run(&img)?, w, h))
                    })
                    .map_err(map_err)?;
                Ok(from_core(r, w, h))
            }
        }
    }

    /// Detection only (boxes carry no text).
    fn run_det_only(&self, py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<OcrOut> {
        match extract_source(image)? {
            Source::Path(p) => {
                let (r, w, h) = py
                    .detach(|| -> Result<_, qp::Error> {
                        let img = qp::decode_file(&p)?;
                        let (w, h) = (img.w, img.h);
                        Ok((self.inner.run_det_only(&img)?, w, h))
                    })
                    .map_err(map_err)?;
                Ok(from_core(r, w, h))
            }
            Source::Bytes(b) => {
                let (r, w, h) = py
                    .detach(|| -> Result<_, qp::Error> {
                        let img = qp::decode_bytes(&b)?;
                        let (w, h) = (img.w, img.h);
                        Ok((self.inner.run_det_only(&img)?, w, h))
                    })
                    .map_err(map_err)?;
                Ok(from_core(r, w, h))
            }
        }
    }

    /// Number of threads the engine pool will use.
    #[staticmethod]
    fn thread_count() -> usize {
        qp::thread_count()
    }

    fn __repr__(&self) -> String {
        "Engine(<qppocr>)".to_string()
    }
}

/// Python module entry point (`import qppocr`).
#[pymodule]
fn qppocr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<OcrOut>()?;
    m.add_class::<Line>()?;
    m.add_class::<Char>()?;
    m.add_class::<Timings>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

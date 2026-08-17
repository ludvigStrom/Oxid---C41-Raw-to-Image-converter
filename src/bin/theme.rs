//! Darker “pro lab” chrome: near-black panels, steel accent, Material Icons.

use std::ops::RangeInclusive;

use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, FontTweak, Rounding, Shadow, Stroke, Vec2,
};

const BG_APP: Color32 = Color32::from_rgb(18, 18, 20);
const BG_PANEL: Color32 = Color32::from_rgb(22, 22, 24);
const BG_WIDGET: Color32 = Color32::from_rgb(32, 32, 36);
const BG_WIDGET_HOVER: Color32 = Color32::from_rgb(42, 42, 48);
const BG_SELECTED: Color32 = Color32::from_rgb(48, 56, 68);
const ACCENT: Color32 = Color32::from_rgb(110, 140, 170);
const TEXT: Color32 = Color32::from_rgb(220, 220, 224);
const TEXT_WEAK: Color32 = Color32::from_rgb(150, 150, 156);
const BORDER: Color32 = Color32::from_rgb(48, 48, 52);
const SCROLL_GRAB: Color32 = Color32::from_rgb(70, 70, 76);

pub const ADD: &str = "\u{e145}";
pub const CLOSE: &str = "\u{e5cd}";
pub const ROTATE_LEFT: &str = "\u{e419}";
pub const ROTATE_RIGHT: &str = "\u{e41a}";
pub const FLIP_H: &str = "\u{e8d4}";
pub const FLIP_V: &str = "\u{e8d5}";
pub const FOLDER: &str = "\u{e2c7}";
pub const UNDO: &str = "\u{e166}";
pub const REDO: &str = "\u{e15a}";
pub const AUTO_FIX: &str = "\u{e663}";
pub const CROP: &str = "\u{e3be}";
pub const DOWNLOAD: &str = "\u{e2c4}";
pub const UPLOAD: &str = "\u{e2c6}";
pub const COLORIZE: &str = "\u{e3b8}";
pub const INVENTORY: &str = "\u{e179}";
pub const EDIT: &str = "\u{e3c9}";
pub const CANCEL: &str = "\u{e5c9}";
pub const ARCHIVE: &str = "\u{e149}";
pub const IMAGE: &str = "\u{e3f4}";
pub const TUNE: &str = "\u{e429}";
pub const EXPOSURE: &str = "\u{e3f6}";
pub const FILTER_HDR: &str = "\u{e3d7}";
pub const CONTRAST: &str = "\u{e3a2}";
pub const WB_AUTO: &str = "\u{e42c}";
pub const PALETTE: &str = "\u{e40a}";
pub const LAYERS: &str = "\u{e53b}";
pub const BLUR: &str = "\u{e3a5}";
pub const MOVIE: &str = "\u{e02c}";
pub const HD: &str = "\u{e052}";

pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "material_icons".to_owned(),
        FontData::from_static(include_bytes!(
            "../../assets/fonts/MaterialSymbolsOutlined.ttf"
        ))
        .tweak(FontTweak {
            y_offset_factor: 0.06,
            ..Default::default()
        }),
    );
    if let Some(family) = fonts.families.get_mut(&FontFamily::Proportional) {
        family.push("material_icons".to_owned());
    }
    if let Some(family) = fonts.families.get_mut(&FontFamily::Monospace) {
        family.push("material_icons".to_owned());
    }
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = visuals();
    style.interaction.selectable_labels = false;
    style.spacing.button_padding = Vec2::new(8.0, 3.0);
    style.spacing.item_spacing = Vec2::new(6.0, 4.0);
    style.spacing.interact_size = Vec2::new(36.0, 18.0);
    let mut scroll = egui::style::ScrollStyle::solid();
    scroll.bar_width = 8.0;
    scroll.handle_min_length = 24.0;
    style.spacing.scroll = scroll;
    ctx.set_style(style);
}

pub fn icon_label(icon: &str, text: &str) -> egui::RichText {
    egui::RichText::new(format!("{icon}  {text}"))
}

pub fn icon_button(ui: &mut egui::Ui, icon: &str, tip: &str) -> egui::Response {
    ui.add(egui::Button::new(egui::RichText::new(icon).size(18.0)).frame(false))
        .on_hover_text(tip)
}

/// Full-width slider row: fixed label, expanding rail, fixed value box.
pub struct SliderRowResponse {
    pub label: egui::Response,
    pub slider: egui::Response,
}

impl SliderRowResponse {
    pub fn changed(&self) -> bool {
        self.slider.changed()
    }
}

/// Label 34% · rail fills the rest · value 18%. Same start x for every row.
pub fn slider_row<N: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    value: &mut N,
    range: RangeInclusive<N>,
    decimals: usize,
) -> SliderRowResponse {
    slider_row_with(ui, label, value, range, |s| s.fixed_decimals(decimals))
}

pub fn slider_row_with<N: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    value: &mut N,
    range: RangeInclusive<N>,
    configure: impl FnOnce(egui::Slider<'_>) -> egui::Slider<'_>,
) -> SliderRowResponse {
    const LABEL_FRACTION: f32 = 0.34;
    const VALUE_FRACTION: f32 = 0.18;

    let gap = ui.spacing().item_spacing.x;
    let row_h = ui.spacing().interact_size.y;
    // Extra right inset so the value box is not clipped by the panel/scrollbar.
    let total = (ui.available_width() - gap).max(0.0);
    let label_w = total * LABEL_FRACTION;
    let value_w = total * VALUE_FRACTION;
    let slider_w = (total - label_w - value_w - gap * 2.0).max(row_h * 2.0);

    let mut label_resp = None;
    let mut slider_resp = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().slider_width = slider_w;
        ui.spacing_mut().interact_size.x = value_w;
        ui.allocate_ui_with_layout(
            egui::vec2(label_w, row_h),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                label_resp = Some(ui.add(egui::Label::new(label).truncate()));
            },
        );
        slider_resp = Some(ui.add(configure(egui::Slider::new(value, range))));
    });

    SliderRowResponse {
        label: label_resp.expect("slider row allocated a label"),
        slider: slider_resp.expect("slider row allocated a slider"),
    }
}

pub fn section_reset(ui: &mut egui::Ui) -> bool {
    ui.add_space(6.0);
    ui.small_button("Reset").clicked()
}

fn visuals() -> egui::Visuals {
    let rounding = Rounding::same(3.0);
    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.window_fill = BG_PANEL;
    v.panel_fill = BG_PANEL;
    v.extreme_bg_color = BG_APP;
    v.faint_bg_color = Color32::from_rgb(26, 26, 28);
    v.code_bg_color = BG_WIDGET;
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = BG_SELECTED;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.window_rounding = rounding;
    v.menu_rounding = rounding;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_shadow = Shadow {
        offset: Vec2::new(4.0, 8.0),
        blur: 12.0,
        spread: 0.0,
        color: Color32::from_black_alpha(80),
    };
    v.popup_shadow = Shadow {
        offset: Vec2::new(2.0, 4.0),
        blur: 8.0,
        spread: 0.0,
        color: Color32::from_black_alpha(70),
    };
    v.slider_trailing_fill = true;
    v.widgets.noninteractive = widget(BG_PANEL, BG_PANEL, BORDER, TEXT_WEAK, 0.0);
    v.widgets.inactive = widget(SCROLL_GRAB, BG_WIDGET, BORDER, TEXT, 0.0);
    v.widgets.hovered = widget(ACCENT, BG_WIDGET_HOVER, ACCENT, TEXT, 1.0);
    v.widgets.active = widget(ACCENT, BG_SELECTED, ACCENT, TEXT, 1.0);
    v.widgets.open = widget(ACCENT, BG_SELECTED, ACCENT, TEXT, 0.0);
    v
}

fn widget(
    bg_fill: Color32,
    weak_bg_fill: Color32,
    stroke: Color32,
    fg: Color32,
    expansion: f32,
) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill,
        weak_bg_fill,
        bg_stroke: Stroke::new(1.0, stroke),
        rounding: Rounding::same(3.0),
        fg_stroke: Stroke::new(1.0, fg),
        expansion,
    }
}

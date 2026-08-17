//! Darker “pro lab” chrome: near-black panels, steel accent, Material Icons.

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

use game_engine::VulkanRenderer;
use raw_window_handle::HasDisplayHandle;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::ElementState;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

struct WindowSettings {
    title: String,
    width: u32,
    height: u32,
}

impl WindowSettings {
    fn new(title: &str, width: u32, height: u32) -> Self {
        WindowSettings {
            title: title.to_string(),
            width,
            height,
        }
    }
}

struct GameEngine {
    window_settings: WindowSettings,
    last_frame: std::time::Instant,
    renderer: Option<VulkanRenderer>,
    egui_context: egui::Context,
    egui_state: Option<egui_winit::State>,
    // window should be the last field to be dropped since egui (and maybe other stuff + future
    // stuff) has an implicit dependancy to it
    window: Option<Arc<Window>>,
}

impl GameEngine {
    fn new(window_settings: WindowSettings) -> GameEngine {
        let egui_context = egui::Context::default();
        GameEngine {
            window: None,
            window_settings,
            last_frame: std::time::Instant::now(),
            renderer: None,
            egui_context,
            egui_state: None,
        }
    }

    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Arc<Window> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(self.window_settings.title.clone())
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        self.window_settings.width,
                        self.window_settings.height,
                    )),
            )
            .expect("Window creation failed");
        let window = Arc::new(window);
        log::info!("succesfully created window");
        window
    }
}

fn handle_egui(window: &Window, ctx: &egui::Context, egui_state: &mut egui_winit::State) {
    let raw_input = egui_state.take_egui_input(window);
    let full_output = ctx.run(raw_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Hello world!");
            if ui.button("Click me").clicked() {
                // take some action here
            }
        });
    });

    egui_state.handle_platform_output(window, full_output.platform_output);
    let clipped_primitives = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    // println!("clipped_primitives: {:?}", clipped_primitives);
    // println!("textures_delta: {:?}", full_output.textures_delta);
    // panic!("AHHH");
    //paint(full_output.textures_delta, clipped_primitives);
}

impl ApplicationHandler for GameEngine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Setting up window and renderer");
        let window = self.init_window(event_loop);

        self.renderer = Some(VulkanRenderer::new(window.clone()));
        let egui_state = egui_winit::State::new(
            self.egui_context.clone(),
            self.egui_context.viewport_id(),
            &window
                .display_handle()
                .expect("Handle should be supported and available"),
            // I hope  this is what they expect me to pass here
            Some(window.scale_factor() as f32),
            window.theme(),
            //4096 x 4096 is hopefully big enough for texture atlas, while still supported by most
            //     hardware
            Some(4096),
        );
        self.egui_state = Some(egui_state);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(renderer), Some(window), Some(egui_state)) = (
            self.renderer.as_mut(),
            self.window.as_ref(),
            self.egui_state.as_mut(),
        ) {
            let event_response = egui_state.on_window_event(window, &event);
            if event_response.repaint {
                log::debug!("Egui wants to be repainted after event \"{:?}\" ", event);
            }
            if event_response.consumed {
                log::debug!("Event \"{:?}\" was consumed by egui", event);
                return;
            }
            let mut exit = false;
            match event {
                WindowEvent::CloseRequested => {
                    log::info!("The close button was pressed; stopping");
                    exit = true;
                }
                WindowEvent::RedrawRequested => {
                    handle_egui(window, &self.egui_context, egui_state);
                    self.last_frame = std::time::Instant::now();
                    window.pre_present_notify();
                    renderer.draw();
                }
                WindowEvent::Resized(physical_size) => {
                    log::warn!(
                        "Resizing not yet implemented. Should resize to {:?}",
                        physical_size
                    );
                    //window_state.resize(physical_size);
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: key,
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => match key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        log::info!("Escape was pressed; Closing window");
                        exit = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        log::info!("Pressing W")
                    }
                    _ => log::info!("Something else was pressed"),
                },
                _ => (),
            }
            if exit {
                event_loop.exit();
                renderer.wait_idle();
            }
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        match cause {
            winit::event::StartCause::Poll => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => log::info!("Ignoring cause: {:?}", cause),
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let window_settings = WindowSettings::new("LexEngine", 800, 600);
    let mut game_engine = GameEngine::new(window_settings);

    event_loop
        .run_app(&mut game_engine)
        .expect("Runtime Error in the eventloop");
    log::info!("Exiting Program");
}

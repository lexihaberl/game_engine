use game_engine::VulkanRenderer;
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
    window: Option<Arc<Window>>,
    window_settings: WindowSettings,
    last_frame: std::time::Instant,
    renderer: Option<VulkanRenderer>,
}

impl GameEngine {
    fn new(window_settings: WindowSettings) -> GameEngine {
        GameEngine {
            window: None,
            window_settings,
            last_frame: std::time::Instant::now(),
            renderer: None,
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

impl ApplicationHandler for GameEngine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Setting up window and renderer");
        let window = self.init_window(event_loop);

        self.renderer = Some(VulkanRenderer::new(window.clone()));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(renderer), Some(window)) = (self.renderer.as_mut(), self.window.as_ref()) {
            let mut exit = false;
            match event {
                WindowEvent::CloseRequested => {
                    log::info!("The close button was pressed; stopping");
                    exit = true;
                }
                WindowEvent::RedrawRequested => {
                    self.last_frame = std::time::Instant::now();
                    window.pre_present_notify();
                    renderer.draw();
                }
                WindowEvent::Resized(physical_size) => {
                    let logical_size = physical_size.to_logical(window.scale_factor());
                    renderer.resize_swapchain(logical_size);
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
                    _ => log::debug!("Something else was pressed"),
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
            _ => log::warn!("Ignoring cause: {:?}", cause),
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let window_settings = WindowSettings::new("LexEngine", 1800, 1000);
    let mut game_engine = GameEngine::new(window_settings);

    event_loop
        .run_app(&mut game_engine)
        .expect("Runtime Error in the eventloop");
    log::info!("Exiting Program");
}

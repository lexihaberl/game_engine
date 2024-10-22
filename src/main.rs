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

struct WindowState {
    window: Arc<Window>,
}

impl WindowState {
    fn new(event_loop: &ActiveEventLoop, settings: &WindowSettings) -> Self {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(settings.title.clone())
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        settings.width,
                        settings.height,
                    )),
            )
            .expect("Window creation failed");
        let window = Arc::new(window);
        log::info!("succesfully created window and renderer");
        WindowState { window }
    }

    fn draw(&mut self) {
        self.window.pre_present_notify();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {}

    fn wait_idle(&self) {}

    fn request_redraw(&self) {}
}

struct GameEngine {
    window_state: Option<WindowState>,
    default_window_settings: WindowSettings,
    last_frame: std::time::Instant,
}

impl GameEngine {
    fn new(default_window_settings: WindowSettings) -> GameEngine {
        GameEngine {
            window_state: None,
            default_window_settings,
            last_frame: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for GameEngine {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window_state = Some(WindowState::new(event_loop, &self.default_window_settings));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(window_state) = &mut self.window_state {
            let mut exit = false;
            match event {
                WindowEvent::CloseRequested => {
                    log::info!("The close button was pressed; stopping");
                    exit = true;
                }
                WindowEvent::RedrawRequested => {
                    self.last_frame = std::time::Instant::now();
                    window_state.draw();
                }
                WindowEvent::Resized(physical_size) => {
                    window_state.resize(physical_size);
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
                window_state.wait_idle();
            }
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        match cause {
            winit::event::StartCause::Poll => {
                if let Some(window_state) = &self.window_state {
                    window_state.request_redraw();
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

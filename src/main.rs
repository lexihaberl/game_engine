use game_engine::VulkanRenderer;
use nalgebra_glm as glm;
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

#[derive(Debug)]
struct Camera {
    velocity: glm::Vec3,
    position: glm::Vec3,
    pitch: f32,
    yaw: f32,
    last_cursor_position: Option<(f64, f64)>,
    vel_sensitivity: f32,
    rot_sensitivity: f32,
}

impl Camera {
    fn new() -> Self {
        Camera {
            velocity: glm::vec3(0.0, 0.0, 0.0),
            position: glm::vec3(0.0, 0.0, 5.0),
            pitch: 0.0,
            yaw: 0.0,
            last_cursor_position: None,
            vel_sensitivity: 0.05,
            rot_sensitivity: 1.0 / 400.0,
        }
    }

    fn get_rotation_matrix(&self) -> glm::Mat4 {
        let pitch_rotation = glm::quat_angle_axis(self.pitch, &glm::vec3(1.0, 0.0, 0.0));
        let yaw_rotation = glm::quat_angle_axis(self.yaw, &glm::vec3(0.0, -1.0, 0.0));
        glm::quat_to_mat4(&pitch_rotation) * glm::quat_to_mat4(&yaw_rotation)
    }

    fn get_view_matrix(&self) -> glm::Mat4 {
        let camera_translation = glm::translate(&glm::Mat4::identity(), &self.position);
        let camera_rotation = self.get_rotation_matrix();
        glm::inverse(&(camera_translation * camera_rotation))
    }

    fn update(&mut self) {
        let rot_mat = self.get_rotation_matrix();
        let pos_vec4 = rot_mat
            * glm::vec4(
                self.velocity.x * self.vel_sensitivity,
                self.velocity.y * self.vel_sensitivity,
                self.velocity.z * self.vel_sensitivity,
                0.0,
            );
        self.position += glm::vec3(pos_vec4.x, pos_vec4.y, pos_vec4.z);
    }

    fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: key,
                        state: element_state,
                        ..
                    },
                ..
            } => match (key, element_state) {
                (PhysicalKey::Code(KeyCode::KeyW), ElementState::Released) => {
                    self.velocity.z = 0.0;
                }
                (PhysicalKey::Code(KeyCode::KeyW), ElementState::Pressed) => {
                    self.velocity.z = -1.0;
                }

                (PhysicalKey::Code(KeyCode::KeyA), ElementState::Released) => {
                    self.velocity.x = 0.0;
                }
                (PhysicalKey::Code(KeyCode::KeyA), ElementState::Pressed) => {
                    self.velocity.x = -1.0;
                }
                (PhysicalKey::Code(KeyCode::KeyS), ElementState::Released) => {
                    self.velocity.z = 0.0;
                }

                (PhysicalKey::Code(KeyCode::KeyS), ElementState::Pressed) => {
                    self.velocity.z = 1.0;
                }
                (PhysicalKey::Code(KeyCode::KeyD), ElementState::Released) => {
                    self.velocity.x = 0.0;
                }
                (PhysicalKey::Code(KeyCode::KeyD), ElementState::Pressed) => {
                    self.velocity.x = 1.0;
                }
                _ => log::debug!("Something else was pressed"),
            },
            //TODO: use deviceevents for raw input instead
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = match self.last_cursor_position {
                    Some(last_position) => {
                        let (last_x, last_y) = last_position;
                        let dx = position.x - last_x;
                        let dy = position.y - last_y;
                        (dx, dy)
                    }
                    None => (0.0, 0.0),
                };
                self.last_cursor_position = Some((position.x, position.y));
                self.yaw += dx as f32 * self.rot_sensitivity;
                self.pitch -= dy as f32 * self.rot_sensitivity;
            }
            _ => (),
        }
    }
}

struct GameEngine<'a> {
    window: Option<Arc<Window>>,
    window_settings: WindowSettings,
    last_frame: std::time::Instant,
    renderer: Option<VulkanRenderer<'a>>,
    camera: Camera,
}

impl<'a> GameEngine<'a> {
    fn new(window_settings: WindowSettings) -> GameEngine<'a> {
        GameEngine {
            window: None,
            window_settings,
            last_frame: std::time::Instant::now(),
            renderer: None,
            camera: Camera::new(),
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

impl<'a> ApplicationHandler for GameEngine<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Setting up window and renderer");
        let window = self.init_window(event_loop);

        self.renderer = Some(VulkanRenderer::new(window.clone()));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(renderer), Some(window)) = (self.renderer.as_mut(), self.window.as_ref()) {
            self.camera.process_event(&event);
            let mut exit = false;
            match event {
                WindowEvent::CloseRequested => {
                    log::info!("The close button was pressed; stopping");
                    exit = true;
                }
                WindowEvent::RedrawRequested => {
                    self.last_frame = std::time::Instant::now();
                    window.pre_present_notify();
                    self.camera.update();

                    renderer.draw(self.camera.get_view_matrix());
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

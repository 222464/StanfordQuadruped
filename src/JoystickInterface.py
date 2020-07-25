import numpy as np
import time
from src.State import BehaviorState, State
from src.Command import Command
from src.Utilities import deadband, clipped_first_order_filter
import evdev
from evdev import list_devices, ecodes

class JoystickInterface:
    def __init__(
        self, config,
    ):
        self.config = config
        self.previous_gait_toggle = 0
        self.previous_state = BehaviorState.REST
        self.previous_hop_toggle = 0
        self.previous_activate_toggle = 0

        self.message_rate = 50

        self.device = evdev.InputDevice(list_devices()[0])
        print(self.device)

        self.ls = [ 0.0, 0.0 ]
        self.rs = [ 0.0, 0.0 ]
        self.l1 = False
        self.r1 = False
        self.y = False
        self.b = False
        self.a = False
        self.x = False
        self.dpad_x = 0
        self.dpad_y = 0

    def get_command(self, state, do_print=False):
        try:
            for event in self.device.read():
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:
                        self.ls[0] = (event.value - 32767.0) / 32768.0
                    elif event.code == ecodes.ABS_Y:
                        self.ls[1] = (event.value - 32767.0) / 32768.0
                    elif event.code == ecodes.ABS_RX:
                        self.rs[0] = (event.value - 32767.0) / 32768.0
                    elif event.code == ecodes.ABS_RY:
                        self.rs[1] = (event.value - 32767.0) / 32768.0
                    elif event.code == 16:
                        self.dpad_x = event.value
                    elif event.code == 17:
                        self.dpad_y = event.value
                elif event.type == ecodes.EV_KEY:
                    if event.code == 306: # Code for Y
                        self.y = event.value != 0
                    if event.code == 304: # Code for B
                        self.b = event.value != 0
                    if event.code == 305: # Code for A
                        self.a = event.value != 0
                    if event.code == 307: # Code for X
                        self.x = event.value != 0
                    elif event.code == 308: # Code for L1
                        self.l1 = event.value != 0
                    elif event.code == 309: # Code for R1
                        self.r1 = event.value != 0
            
                #print(event)
        except:
            pass

        assert(self.ls[0] >= -1.0 and self.ls[0] <= 1.0)
        assert(self.ls[1] >= -1.0 and self.ls[1] <= 1.0)
        assert(self.rs[0] >= -1.0 and self.rs[0] <= 1.0)
        assert(self.rs[1] >= -1.0 and self.rs[1] <= 1.0)

        command = Command()
        
        ####### Handle discrete commands ########
        # Check if requesting a state transition to trotting, or from trotting to resting
        gait_toggle = self.r1
        command.trot_event = (gait_toggle == 1 and self.previous_gait_toggle == 0)

        # Check if requesting a state transition to hopping, from trotting or resting
        hop_toggle = self.x
        command.hop_event = (hop_toggle == 1 and self.previous_hop_toggle == 0)            
        
        activate_toggle = self.l1
        command.activate_event = (activate_toggle == 1 and self.previous_activate_toggle == 0)

        # Update previous values for toggles and state
        self.previous_gait_toggle = gait_toggle
        self.previous_hop_toggle = hop_toggle
        self.previous_activate_toggle = activate_toggle

        ####### Handle continuous commands ########
        x_vel = self.ls[1] * -self.config.max_x_velocity
        y_vel = self.ls[0] * -self.config.max_y_velocity
        command.horizontal_velocity = np.array([x_vel, y_vel])
        command.yaw_rate = self.rs[0] * -self.config.max_yaw_rate

        message_rate = self.message_rate
        message_dt = 1.0 / message_rate

        pitch = self.rs[1] * self.config.max_pitch
        deadbanded_pitch = deadband(
            pitch, self.config.pitch_deadband
        )
        pitch_rate = clipped_first_order_filter(
            state.pitch,
            deadbanded_pitch,
            self.config.max_pitch_rate,
            self.config.pitch_time_constant,
        )
        command.pitch = state.pitch + message_dt * pitch_rate

        height_movement = self.dpad_y
        command.height = state.height - message_dt * self.config.z_speed * height_movement
        
        roll_movement = -self.dpad_x
        command.roll = state.roll + message_dt * self.config.roll_speed * roll_movement

        return command

    def set_color(self, color):
        # joystick_msg = {"ps4_color": color}
        # self.udp_publisher.send(joystick_msg)
        pass
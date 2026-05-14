import unittest
import sys
import types


scrcpy_stub = types.SimpleNamespace(ACTION_DOWN=0, ACTION_MOVE=1, ACTION_UP=2)
sys.modules.setdefault("scrcpy", scrcpy_stub)
sys.modules.setdefault("adbutils", types.SimpleNamespace(adb=types.SimpleNamespace(device_list=lambda: [])))

from window_controller import WindowController


class WindowControllerInputTests(unittest.TestCase):
    def make_controller(self):
        wc = WindowController.__new__(WindowController)
        wc.are_we_moving = True
        wc.last_joystick_pos = (10, 20)
        wc.last_joystick_keepalive_time = 0.0
        wc.joystick_debug = False
        wc.PID_JOYSTICK = 1
        wc.PID_ATTACK = 2
        wc.width_ratio = 1.0
        wc.height_ratio = 1.0
        wc.scale_factor = 1.0
        wc.moves = []
        wc.touches = []
        wc.touch_move = lambda x, y, pointer_id=0: wc.moves.append((x, y, pointer_id))
        wc.touch_down = lambda x, y, pointer_id=0: wc.touches.append(("down", x, y, pointer_id))
        wc.touch_up = lambda x, y, pointer_id=0: wc.touches.append(("up", x, y, pointer_id))
        return wc

    def test_restore_movement_after_attack_replays_last_joystick_position(self):
        wc = self.make_controller()

        self.assertTrue(wc.restore_movement_after_attack(before_vector=(10, 20)))

        self.assertEqual(wc.moves[-1], (10, 20, 1))

    def test_restore_movement_after_attack_noops_when_not_moving(self):
        wc = self.make_controller()
        wc.are_we_moving = False

        self.assertFalse(wc.restore_movement_after_attack())
        self.assertEqual(wc.moves, [])

    def test_aim_attack_angle_keeps_movement_when_parallel_release_not_forced(self):
        wc = self.make_controller()
        pause_calls = []
        wc.pause_movement_for_attack = lambda: pause_calls.append("pause")

        wc.aim_attack_angle(0.0, duration=0.0, force_release_movement=False)

        self.assertEqual(pause_calls, [])
        self.assertTrue(wc.are_we_moving)
        self.assertEqual(wc.moves[-1], (10, 20, 1))
        self.assertIn(("down", 1725.0, 800.0, 2), wc.touches)

    def test_aim_attack_angle_pause_resume_when_release_forced(self):
        wc = self.make_controller()
        pause_calls = []
        resume_calls = []

        def pause():
            pause_calls.append("pause")
            return (10, 20)

        def resume(movement_before):
            resume_calls.append(movement_before)
            return True

        wc.pause_movement_for_attack = pause
        wc.resume_movement_after_attack = resume

        wc.aim_attack_angle(0.0, duration=0.0, force_release_movement=True)

        self.assertEqual(pause_calls, ["pause"])
        self.assertEqual(resume_calls, [(10, 20)])


if __name__ == "__main__":
    unittest.main()

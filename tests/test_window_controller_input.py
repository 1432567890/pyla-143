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
        wc.moves = []
        wc.touch_move = lambda x, y, pointer_id=0: wc.moves.append((x, y, pointer_id))
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


if __name__ == "__main__":
    unittest.main()

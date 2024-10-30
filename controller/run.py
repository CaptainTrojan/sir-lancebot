import argparse
from controller import BotController

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Arena dimensions")
    parser.add_argument("width", type=int, help="Width of the arena (mm)")
    parser.add_argument("length", type=int, help="Length of the arena (mm)")
    parser.add_argument("bot_width", type=int, help="Width of the bot (mm)")
    parser.add_argument("bot_length", type=int, help="Length of the bot (mm)")
    parser.add_argument("aruco_size", type=int, help="Size of the ArUco markers (mm)")
    parser.add_argument("aruco_margin", type=int, help="Margin between the ArUco markers and the bot edge (mm)")
    parser.add_argument("-pw", "--preview_width", type=int, default=900, help="Width of the preview windows")
    args = parser.parse_args()

    # Create the controller
    controller = BotController(args)
    
    # Run the controller
    controller.run()
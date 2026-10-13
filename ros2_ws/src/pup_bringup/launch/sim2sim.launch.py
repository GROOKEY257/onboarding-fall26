"""Bring up the whole sim2sim stack: simulator + policy (+ optional teleop).

    ros2 launch pup_bringup sim2sim.launch.py headless:=false teleop:=true
    ros2 launch pup_bringup sim2sim.launch.py policy_path:=/ws/runs/colab/policy.npz

`pup_sim/launch/sim_only.launch.py` is the template this is built from.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Read values supplied when running ros2 launch.
    headless = LaunchConfiguration("headless")
    policy_path = LaunchConfiguration("policy_path")
    teleop = LaunchConfiguration("teleop")
    realtime_factor = LaunchConfiguration("realtime_factor")

    return LaunchDescription([
        # Declare launch arguments.
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description="Run the simulator without a graphical window",
        ),
        DeclareLaunchArgument(
            "policy_path",
            default_value="/ws/pup/checkpoints/pup_joystick_flat_reference.npz",
            description="Path to the exported NumPy policy",
        ),
        DeclareLaunchArgument(
            "teleop",
            default_value="false",
            description="Launch the teleoperation node",
        ),
        DeclareLaunchArgument(
            "realtime_factor",
            default_value="1.0",
            description="Simulator real-time factor",
        ),

        # Launch the MuJoCo simulator.
        Node(
            package="pup_sim",
            executable="sim_node",
            name="pup_sim",
            output="screen",
            parameters=[{
                "headless": headless,
                "realtime_factor": realtime_factor,
            }],
        ),

        # Launch the policy controller.
        Node(
            package="pup_bringup",
            executable="policy_node",
            name="pup_policy",
            output="screen",
            parameters=[{
                "policy_path": policy_path,
            }],
        ),

        # Launch teleoperation only when teleop:=true.
        Node(
            package="pup_sim",
            executable="teleop_node",
            name="pup_teleop",
            output="screen",
            condition=IfCondition(teleop),
        ),
    ])

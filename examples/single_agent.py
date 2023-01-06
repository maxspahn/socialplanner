import gym
from pathlib import Path
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.goalComposition import GoalComposition
from socialplanner.social_planner import SocialPlanner
"""
Fabrics example for a 3D point mass robot.
The fabrics planner uses a 2D point mass to compute actions for a simulated 3D point mass.

To do: tune behavior.
"""

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    initial_observation = env.reset(pos=pos0, vel=vel0)
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 1000.0, 0.0], "radius": 1.0},
    }
    static_obst = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    dynamic_obst_dict = {
        "type": "analyticSphere",
        "geometry": {"trajectory": ["300 - 0.0 * t", "2.0 * ca.sin(1 * t)"], "radius": 0.4},
    }
    dynamic_obst = DynamicSphereObstacle(name="staticObst", content_dict=dynamic_obst_dict)
    obstacles = (static_obst, dynamic_obst) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # Add walls, the goal and the obstacle to the environment.
    #env.add_walls([0.1, 10, 0.5], [[5.0, 0, 0], [-5.0, 0.0, 0.0], [0.0, 5.0, np.pi/2], [0.0, -5.0, np.pi/2]])
    env.add_goal(goal)
    env.add_obstacle(static_obst)
    env.add_obstacle(dynamic_obst)
    return (env, obstacles, goal, initial_observation)


def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    initial_state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0.0, 0.0],
            # [0.5, 10, -0.5, -0.5, 0.5, 0.0],
            # [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )
    # social groups informoation is represented as lists of indices of the state array
    # groups = [[1, 0], [2]]
    groups = []
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    # obs = [[1, 2, 7, 8]]
    obs = []

    planner = SocialPlanner(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("example.toml"),
    )

    return planner


def run_point_robot_urdf(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    
    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, obstacles, goal, initial_observation) = initalize_environment(render)
    ob = initial_observation
    obst1, dynamic_obstacle = obstacles
    print(f"Initial observation : {ob}")
    action = np.array([0.0, 0.0, 0.0])
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = np.array(goal.sub_goals()[0].weight())
    obst1_position = np.array(obst1.position())
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        position_dynamic_obstacle = dynamic_obstacle.position(t=env.t())
        velocity_dynamic_obstacle = dynamic_obstacle.velocity(t=env.t())
        acceleration_dynamic_obstacle = dynamic_obstacle.acceleration(t=env.t()) * 0
        action[0:2] = planner.compute_action(
            q=ob["robot_0"]["joint_state"]["position"][0:2],
            qdot=ob["robot_0"]["joint_state"]["velocity"][0:2],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst1_position[0:2],
            radius_obst_0=np.array([obst1.radius()]),
            radius_body_1=np.array([0.2]),
            radius_dynamic_obst_0=dynamic_obstacle.radius(),
            x_ref_dynamic_obst_0_1_leaf=position_dynamic_obstacle,
            xdot_ref_dynamic_obst_0_1_leaf=velocity_dynamic_obstacle,
            xddot_ref_dynamic_obst_0_1_leaf=acceleration_dynamic_obstacle,
        )
        ob, *_, = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)

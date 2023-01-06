from copy import deepcopy
from pathlib import Path

import gym
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

number_agents = 9

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
    robots = []
    for _ in range(number_agents):
        robots.append(GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"))
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.ones((number_agents, 3)) * -5 + np.random.rand(number_agents, 3) * 10
    vel0 = np.ones((number_agents, 3)) * -1 + np.random.rand(number_agents, 3) * 2
    vel0[:,2] = 0
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
                'low': [-5, -5],
                'high': [5, 5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal_template = GoalComposition(name="goal", content_dict=goal_dict)
    goals = []
    for _ in range(number_agents):
        goal_i = deepcopy(goal_template)
        goal_i.shuffle()
        env.add_goal(goal_i)
        goals.append(goal_i)
    env.add_obstacle(static_obst)
    env.add_obstacle(dynamic_obst)
    return (env, obstacles, goals, initial_observation)


def set_planner(initial_observation):
    """
    Initializes the social forces planner for the point robot.

    """
    # social groups informoation is represented as lists of indices of the state array
    # groups = [[1, 0], [2]]
    groups = []
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    # obs = [[1, 2, 7, 8]]
    obs = []
    initial_state = np.zeros((number_agents, 6))
    for i in range(number_agents):
        initial_state[i, :] = np.concatenate((
            initial_observation[f'robot_{i}']['joint_state']['position'][0:2],
            initial_observation[f'robot_{i}']['joint_state']['velocity'][0:2],
            np.zeros(2),
    ))

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
    (env, obstacles, goals, initial_observation) = initalize_environment(render)
    ob = initial_observation
    obst1, dynamic_obstacle = obstacles
    print(f"Initial observation : {ob}")
    action = np.zeros(number_agents * 3)
    planner = set_planner(initial_observation)
    # Start the simulation.
    print("Starting simulation")
    obst1_position = np.array(obst1.position())
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        position_dynamic_obstacle = dynamic_obstacle.position(t=env.t())
        velocity_dynamic_obstacle = dynamic_obstacle.velocity(t=env.t())
        acceleration_dynamic_obstacle = dynamic_obstacle.acceleration(t=env.t()) * 0
        group_action = planner.compute_action(
            ob=ob,
            x_goal_0=goals[0].sub_goals()[0].position(),
            x_goal_1=goals[1].sub_goals()[0].position(),
            x_goal_2=goals[2].sub_goals()[0].position(),
            x_goal_3=goals[3].sub_goals()[0].position(),
            x_goal_4=goals[4].sub_goals()[0].position(),
            x_goal_5=goals[5].sub_goals()[0].position(),
            x_goal_6=goals[6].sub_goals()[0].position(),
            x_goal_7=goals[7].sub_goals()[0].position(),
            x_goal_8=goals[8].sub_goals()[0].position(),
            x_obst_0=obst1_position[0:2],
            radius_obst_0=np.array([obst1.radius()]),
            radius_body_1=np.array([0.2]),
            radius_dynamic_obst_0=dynamic_obstacle.radius(),
            x_ref_dynamic_obst_0_1_leaf=position_dynamic_obstacle,
            xdot_ref_dynamic_obst_0_1_leaf=velocity_dynamic_obstacle,
            xddot_ref_dynamic_obst_0_1_leaf=acceleration_dynamic_obstacle,
        )
        for i in range(number_agents):
            action[3 * i + 0: 3 * i + 2] = group_action[i]
        ob, *_, = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)

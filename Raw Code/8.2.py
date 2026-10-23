#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 2 00:00:00 2026

@author: Andy.
"""

#%%

import numpy as np
import math
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import pandas as pd

import mesa
# print(mesa.__version__)
import networkx as nx
from enum import Enum

# %matplotlib inline
# %precision 2

import logging
logging.basicConfig(level = logging.INFO)

import elfi

#%% ONS Smoking Data: 1974-2023, interpolated to yearly updates

observed_data = pd.read_excel("ONS - UK Smoking Data (1974-2023).xlsx", index_col = 0) # Great Britain

interpolated_data = pd.concat([observed_data[:1],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1975]), observed_data[1:]])
interpolated_data = pd.concat([interpolated_data[:3],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1977]), interpolated_data[3:]])
interpolated_data = pd.concat([interpolated_data[:5],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1979]), interpolated_data[5:]])
interpolated_data = pd.concat([interpolated_data[:7],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1981]), interpolated_data[7:]])
interpolated_data = pd.concat([interpolated_data[:9],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1983]), interpolated_data[9:]])
interpolated_data = pd.concat([interpolated_data[:11],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1985]), interpolated_data[11:]])
interpolated_data = pd.concat([interpolated_data[:13],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1987]), interpolated_data[13:]])
interpolated_data = pd.concat([interpolated_data[:15],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1989]), interpolated_data[15:]])
interpolated_data = pd.concat([interpolated_data[:17],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1991]), interpolated_data[17:]])
interpolated_data = pd.concat([interpolated_data[:19],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1993]), interpolated_data[19:]])
interpolated_data = pd.concat([interpolated_data[:21],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1995]), interpolated_data[21:]])
interpolated_data = pd.concat([interpolated_data[:23],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1997]), interpolated_data[23:]])
interpolated_data = pd.concat([interpolated_data[:25],
                           pd.DataFrame({'% NEVERSMOKER': [np.nan], '% SMOKER': [np.nan], '% QUITTER': [np.nan]},
                                        index=[1999]), interpolated_data[25:]])

interpolated_data['% NEVERSMOKER'] = interpolated_data['% NEVERSMOKER'].interpolate(method = 'linear')
interpolated_data['% SMOKER'] = interpolated_data['% SMOKER'].interpolate(method = 'linear')
interpolated_data['% QUITTER'] = interpolated_data['% QUITTER'].interpolate(method = 'linear')

#%% ONS Population by Age Data: 1974-2024

# population_by_age_data = pd.read_excel("ONS - UK Data for Population by Age (1974-2024).xlsx", index_col = 0) # Great Britain
population_by_age_data = pd.read_excel("ONS - UK Data for Population by Age (1974-2024).xlsx") # Great Britain

# Make sure the year column is called "year"
population_by_age_data = population_by_age_data.rename(columns = {population_by_age_data.columns[0]: "Year"})

# Use year as the index
population_by_age_data = population_by_age_data.set_index("Year")

# Convert all column names to strings and rename 100 to 100+
population_by_age_data.columns = population_by_age_data.columns.map(
    lambda col: "100+" if str(col) == "100" else str(col)
)

#%% ONS Mortality Rates by Age Data - NEVERSMOKER: 

mortality_by_age_data_NEVERSMOKER = pd.read_excel("ONS - Mortality Rates by Age Data (1974-2023) Updated - NEVERSMOKER.xlsx")

# Assume first column is year
mortality_by_age_data_NEVERSMOKER = mortality_by_age_data_NEVERSMOKER.rename(columns = {mortality_by_age_data_NEVERSMOKER.columns[0]: "Year"})

# Use year as index
mortality_by_age_data_NEVERSMOKER = mortality_by_age_data_NEVERSMOKER.set_index("Year")

# Convert age columns to strings so 100+ is handled consistently
mortality_by_age_data_NEVERSMOKER.columns = mortality_by_age_data_NEVERSMOKER.columns.astype(str)

#%% ONS Mortality Rates by Age Data - SMOKER: 

mortality_by_age_data_SMOKER = pd.read_excel("ONS - Mortality Rates by Age Data (1974-2023) Updated - SMOKER.xlsx")

# Assume first column is year
mortality_by_age_data_SMOKER = mortality_by_age_data_SMOKER.rename(columns = {mortality_by_age_data_SMOKER.columns[0]: "Year"})

# Use year as index
mortality_by_age_data_SMOKER = mortality_by_age_data_SMOKER.set_index("Year")

# Convert age columns to strings so 100+ is handled consistently
mortality_by_age_data_SMOKER.columns = mortality_by_age_data_SMOKER.columns.astype(str)

#%% ONS Mortality Rates by Age Data - QUITTER: 

mortality_by_age_data_QUITTER = pd.read_excel("ONS - Mortality Rates by Age Data (1974-2023) Updated - QUITTER.xlsx")

# Assume first column is year
mortality_by_age_data_QUITTER = mortality_by_age_data_QUITTER.rename(columns = {mortality_by_age_data_QUITTER.columns[0]: "Year"})

# Use year as index
mortality_by_age_data_QUITTER = mortality_by_age_data_QUITTER.set_index("Year")

# Convert age columns to strings so 100+ is handled consistently
mortality_by_age_data_QUITTER.columns = mortality_by_age_data_QUITTER.columns.astype(str)

#%% ONS Population turning Age 16 Data: 1974-2024 (based on Crude Birth Rate)

crude_introduction_data = pd.read_excel("ONS - UK Data for Population turning Age 16 by Year (1974-2024).xlsx", index_col = 0)

#%% ONS Population turning Age 16 Data: 1974-2024 (based on Crude Birth Rate and Net Migration)

mig_inc_introduction_data = pd.read_excel("ONS - UK Data for Population turning Age 16 by Year (1974-2024) - Migration Included.xlsx", index_col = 0)

#%%

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize = (9, 6), dpi = 1000)

plt.plot(
    crude_introduction_data.index,
    crude_introduction_data.iloc[:, 0],
    marker="o",
    linestyle="-",
    color="black",
    label="Crude rate"
)

plt.plot(
    mig_inc_introduction_data.index,
    mig_inc_introduction_data.iloc[:, 0],
    marker="o",
    linestyle="-",
    color="red",
    label="Rate including net migration"
)

plt.title("Population Turning Age 16: Crude versus Net Migration Included")
plt.xlabel("Year")
plt.ylabel("Turning age 16 rate per 1000")
plt.grid(True, linestyle="--", alpha=0.5)

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{int(x):,}")
)

plt.legend()
plt.tight_layout()
plt.show()

#%%

# ----------

#%% Smoking states

class State(Enum):
    """Define the possible smoking states for agents."""
    
    NEVERSMOKER = 0
    SMOKER = 1
    QUITTER = 2
    DEAD = 3

#%% Agent-Based Model using existing optimised network

class NSQ_Model_UK(mesa.Model):
    """NSQ model using an existing optimised network."""

    def __init__(
            self,
            # Year to start/end
            year_to_start, # 1974
            year_to_end, # 2023

            # Initial population parameters
            initial_NEVERSMOKER_pct, # 37.4% in 1974
            initial_SMOKER_pct, # 45.6% in 1974
            initial_QUITTER_pct, # 17.0% in 1974

            # Parameters to be calibrated
            delta_N_to_S,
            delta_S_to_Q,
            delta_Q_to_S,
            beta_S_to_Q_due_to_N,
            beta_Q_to_S_due_to_S,

            # Existing optimised network
            input_graph,
            input_node_age=None,

            # Calibrated parameters
            beta_N_to_S_due_to_S=0.40719,
            beta_S_to_Q_due_to_Q=0.35214,

            # Optional seed
            seed=None
            ):
        """Initialise the NSQ_Model_UK class."""

        # Initialise Mesa model
        super().__init__(seed=seed)

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        self.delta_N_to_S = delta_N_to_S
        self.delta_S_to_Q = delta_S_to_Q
        self.delta_Q_to_S = delta_Q_to_S

        self.beta_N_to_S_due_to_S = beta_N_to_S_due_to_S
        self.beta_S_to_Q_due_to_N = beta_S_to_Q_due_to_N
        self.beta_S_to_Q_due_to_Q = beta_S_to_Q_due_to_Q
        self.beta_Q_to_S_due_to_S = beta_Q_to_S_due_to_S

        self.beta_N_to_S_due_to_S_est = (
            1 - (
                1 - beta_N_to_S_due_to_S * (1 - (1 - delta_N_to_S) ** 4.5)
            ) ** (1 / 4.5)
        )

        self.beta_S_to_Q_due_to_Q_est = (
            1 - (
                1 - beta_S_to_Q_due_to_Q * (1 - (1 - delta_S_to_Q) ** 4.5)
            ) ** (1 / 4.5)
        )

        self.year_to_start = year_to_start
        self.year_to_end = year_to_end
        self.current_year = year_to_start
        self.step_count = 0

        self.initial_NEVERSMOKER_pct = initial_NEVERSMOKER_pct
        self.initial_SMOKER_pct = initial_SMOKER_pct
        self.initial_QUITTER_pct = initial_QUITTER_pct

        initial_sum = (
            initial_NEVERSMOKER_pct
            + initial_SMOKER_pct
            + initial_QUITTER_pct
        )

        if initial_sum > 1 + 1e-12:
            raise ValueError(
                "Initial NEVERSMOKER, SMOKER, and QUITTER percentages sum to more than 1."
            )

        self.initial_DEAD_pct = max(0.0, 1 - initial_sum)

        # --------------------------------------------------
        # Use existing optimised network
        # --------------------------------------------------

        self.graph = input_graph.copy()

        # If node ages are provided separately, attach them to graph
        if input_node_age is not None:
            nx.set_node_attributes(
                self.graph,
                {node: int(age) for node, age in input_node_age.items()},
                "age"
            )

        # Remove any old Mesa agent information if present
        for node in self.graph.nodes():
            self.graph.nodes[node].pop("agent", None)
            self.graph.nodes[node].pop("agent_id", None)
            self.graph.nodes[node].pop("status", None)
            self.graph.nodes[node].pop("status_code", None)

        # Check every node has an age
        missing_age_nodes = [
            node
            for node, attrs in self.graph.nodes(data=True)
            if "age" not in attrs or pd.isna(attrs["age"])
        ]

        if len(missing_age_nodes) > 0:
            raise ValueError(
                f"Some nodes do not have an age attribute. "
                f"Example missing nodes: {missing_age_nodes[:10]}"
            )

        # Ensure node ages are integers and store initial degree
        for node in self.graph.nodes():
            self.graph.nodes[node]["age"] = int(self.graph.nodes[node]["age"])
            self.graph.nodes[node]["initial_age"] = int(self.graph.nodes[node]["age"])
            self.graph.nodes[node]["initial_degree"] = int(self.graph.degree[node])

        # Initialise Mesa NetworkGrid using the existing graph
        self.grid = mesa.space.NetworkGrid(self.graph)

        # Next node ID for future introduced adults
        if len(self.graph.nodes()) > 0:
            self.next_node_id = max(self.graph.nodes()) + 1
        else:
            self.next_node_id = 0

        # --------------------------------------------------
        # Add one agent to each existing node
        # Age comes from the existing network node's age
        # --------------------------------------------------

        for node in self.graph.nodes():

            assigned_age = int(self.graph.nodes[node]["age"])

            agent = NSQ_Agent(
                self,
                assigned_age,
                State.NEVERSMOKER
            )

            self.grid.place_agent(agent, node)

            self.set_agent_state(agent, State.NEVERSMOKER)

        # Assign initial smoking statuses only
        self.assign_initial_statuses()

        # --------------------------------------------------
        # Data collector
        # --------------------------------------------------

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "NEVERSMOKER_nodes": lambda m: m.get_nodes_by_state(State.NEVERSMOKER),
                "SMOKER_nodes": lambda m: m.get_nodes_by_state(State.SMOKER),
                "QUITTER_nodes": lambda m: m.get_nodes_by_state(State.QUITTER),
                "DEAD_nodes": lambda m: m.get_nodes_by_state(State.DEAD),

                "NEVERSMOKER_count": lambda m: m.count_agents_by_state(State.NEVERSMOKER),
                "SMOKER_count": lambda m: m.count_agents_by_state(State.SMOKER),
                "QUITTER_count": lambda m: m.count_agents_by_state(State.QUITTER),
                "DEAD_count": lambda m: m.count_agents_by_state(State.DEAD),

                "% NEVERSMOKER": lambda m: m.pct_agents_by_state(State.NEVERSMOKER),
                "% SMOKER": lambda m: m.pct_agents_by_state(State.SMOKER),
                "% QUITTER": lambda m: m.pct_agents_by_state(State.QUITTER),
                "% DEAD": lambda m: m.pct_agents_by_state(State.DEAD)
            },
            agent_reporters={
                "Age": lambda a: a.current_age,
                "State": lambda a: a.current_state,
                "Agent": lambda a: a.unique_id,
                "Node": lambda a: a.pos
            }
        )

        # Capture data from initial setting
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Initial state assignment helpers
    # ------------------------------------------------------------------

    def set_agent_state(self, agent, state):
        """
        Set smoking state for both the agent and the corresponding graph node.
        """

        agent.current_state = state
        agent.updated_state = state

        if agent.pos is not None and agent.pos in self.graph.nodes:
            self.graph.nodes[agent.pos]["status"] = state.name
            self.graph.nodes[agent.pos]["status_code"] = state.value
            self.graph.nodes[agent.pos]["age"] = int(agent.current_age)
            self.graph.nodes[agent.pos]["agent_id"] = agent.unique_id

    def assign_initial_statuses(self):
        """
        Assign initial NEVERSMOKER, SMOKER, QUITTER, DEAD states.
        Ages are not assigned here; ages come from the input network.
        """

        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)

        nodes_shuffled = nodes.copy()
        self.random.shuffle(nodes_shuffled)

        n_smoker = min(
            round(self.initial_SMOKER_pct * n_nodes),
            n_nodes
        )

        n_quitter = min(
            round(self.initial_QUITTER_pct * n_nodes),
            n_nodes - n_smoker
        )

        n_dead = min(
            round(self.initial_DEAD_pct * n_nodes),
            n_nodes - n_smoker - n_quitter
        )

        smoker_nodes = nodes_shuffled[:n_smoker]

        quitter_nodes = nodes_shuffled[
            n_smoker:n_smoker + n_quitter
        ]

        dead_nodes = nodes_shuffled[
            n_smoker + n_quitter:n_smoker + n_quitter + n_dead
        ]

        for node in smoker_nodes:
            agent = self.get_agent_at_node(node)
            if agent is not None:
                self.set_agent_state(agent, State.SMOKER)

        for node in quitter_nodes:
            agent = self.get_agent_at_node(node)
            if agent is not None:
                self.set_agent_state(agent, State.QUITTER)

        for node in dead_nodes:
            agent = self.get_agent_at_node(node)
            if agent is not None:
                self.set_agent_state(agent, State.DEAD)

    # ------------------------------------------------------------------
    # State and count helpers
    # ------------------------------------------------------------------

    def get_agents_by_state(self, state):
        """Return list of agents in a given state."""
        return list(
            filter(
                lambda agent: agent.current_state is state,
                self.agents
            )
        )

    def get_nodes_by_state(self, state):
        """Return list of nodes whose agents are in a given state."""
        return list(
            map(
                lambda agent: agent.pos,
                self.get_agents_by_state(state)
            )
        )

    def count_agents_by_state(self, state):
        """Calculate counts of agents in a given state."""
        return len(self.get_agents_by_state(state))

    def pct_agents_by_state(self, state):
        """Calculate percentage of agents in a given state."""
        return (self.count_agents_by_state(state) / len(self.agents)) * 100

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------

    def mortality_probability(self, agent):
        """Return mortality probability by age and smoking status."""

        if agent.current_age >= 100:
            age_column = "100+"
        else:
            age_column = str(agent.current_age)

        if agent.current_state is State.NEVERSMOKER:
            mortality_probability = float(
                mortality_by_age_data_NEVERSMOKER.loc[
                    self.current_year,
                    age_column
                ]
            )

        elif agent.current_state is State.SMOKER:
            mortality_probability = float(
                mortality_by_age_data_SMOKER.loc[
                    self.current_year,
                    age_column
                ]
            )

        elif agent.current_state is State.QUITTER:
            mortality_probability = float(
                mortality_by_age_data_QUITTER.loc[
                    self.current_year,
                    age_column
                ]
            )

        else:
            mortality_probability = 0.0

        return min(mortality_probability, 1.0)

    def remove_dead(self):
        """Remove agents who die according to age and smoking status."""

        dead_agents_to_remove = []

        for agent in list(self.agents):

            if agent.current_state is State.DEAD:
                continue

            death_probability = self.mortality_probability(agent)

            if self.random.random() < death_probability:
                dead_agents_to_remove.append(agent)

        dead_nodes_to_remove = [
            agent.pos
            for agent in dead_agents_to_remove
            if agent.pos is not None
        ]

        for agent in dead_agents_to_remove:

            self.set_agent_state(agent, State.DEAD)

            self.grid.remove_agent(agent)
            agent.remove()

        self.graph.remove_nodes_from(dead_nodes_to_remove)

        return self.agents, self.graph

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def get_agent_at_node(self, node):
        """Return the single agent located at a node."""

        agents = self.grid.get_cell_list_contents([node])

        if len(agents) == 0:
            return None

        return agents[0]

    def age_fitness(self, age, min_age=16, age_scale=0.67926):
        """
        Fitness decreases with age.

        Larger age_scale = slower decline.
        Smaller age_scale = faster decline.
        """

        age_difference = max(0, age - min_age)

        return 1 / (age_scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (age_difference / age_scale) ** 2)

    def node_attractiveness(
            self,
            node,
            degree_exponent=1.0,
            age_scale=0.67926
            ):
        """
        Bianconi-Barabási-style attractiveness.

        attractiveness = fitness(age) * (degree + 1)^degree_exponent
        """

        agent = self.get_agent_at_node(node)

        if agent is None:
            return 0.0

        if agent.current_state is State.DEAD:
            return 0.0

        degree = self.graph.degree(node)

        degree_component = (degree) ** degree_exponent # degree + 1/0

        fitness_component = self.age_fitness(
            agent.current_age,
            min_age=16,
            age_scale=age_scale
        )

        return fitness_component * degree_component

    def weighted_sample_without_replacement(self, items, weights, k):
        """Sample k items without replacement using weights."""

        selected = []

        items = list(items)
        weights = list(weights)

        k = min(k, len(items))

        for _ in range(k):

            total_weight = sum(weights)

            if total_weight <= 0:
                chosen_index = self.random.randrange(len(items))

            else:
                r = self.random.random() * total_weight
                cumulative = 0

                chosen_index = 0

                for i, weight in enumerate(weights):
                    cumulative += weight

                    if r <= cumulative:
                        chosen_index = i
                        break

            selected.append(items.pop(chosen_index))
            weights.pop(chosen_index)

        return selected

    # ------------------------------------------------------------------
    # Introduce new adults
    # ------------------------------------------------------------------

    def introduce_adult(
            self,
            number_of_BIRTH,
            new_agent_age=16,
            average_degree=4,
            age_scale=0.67926,
            degree_exponent=1.0
            ):
        """
        Introduce new adult agents using age-fitness preferential attachment.
        """

        new_nodes = []

        # --------------------------------------------------
        # Step 1: add new nodes and agents
        # --------------------------------------------------

        for _ in range(number_of_BIRTH):

            new_node = self.next_node_id
            self.next_node_id += 1

            self.graph.add_node(new_node)

            self.graph.nodes[new_node]["age"] = int(new_agent_age)
            self.graph.nodes[new_node]["initial_age"] = int(new_agent_age)
            self.graph.nodes[new_node]["initial_degree"] = 0
            self.graph.nodes[new_node]["status"] = State.NEVERSMOKER.name
            self.graph.nodes[new_node]["status_code"] = State.NEVERSMOKER.value
            self.graph.nodes[new_node]["agent"] = []

            agent = NSQ_Agent(
                self,
                new_agent_age,
                State.NEVERSMOKER
            )

            self.grid.place_agent(agent, new_node)

            self.set_agent_state(agent, State.NEVERSMOKER)

            new_nodes.append(new_node)

        # --------------------------------------------------
        # Step 2: connect each newly added node
        # --------------------------------------------------

        for new_node in new_nodes:

            candidate_nodes = [
                node for node in self.graph.nodes
                if node != new_node
            ]

            weights = [
                self.node_attractiveness(
                    node,
                    degree_exponent=degree_exponent,
                    age_scale=age_scale
                )
                for node in candidate_nodes
            ]

            number_of_connections = min(
                average_degree,
                len(candidate_nodes)
            )

            if number_of_connections <= 0:
                continue

            targets = self.weighted_sample_without_replacement(
                candidate_nodes,
                weights,
                number_of_connections
            )

            for target in targets:
                self.graph.add_edge(new_node, target)

        return self.agents, self.graph

    # ------------------------------------------------------------------
    # Model step and running
    # ------------------------------------------------------------------

    def step(self):
        """Advance model by one step."""

        self.current_year += 1

        # Existing agents update smoking status and age
        self.agents.do("step")
        self.agents.do("advance")

        # Remove agents by mortality rule
        self.remove_dead()

        if len(self.graph.nodes()) > 0:
            self.next_node_id = max(
                self.next_node_id,
                max(self.graph.nodes()) + 1
            )

        # Introduce new adult agents
        number_of_BIRTH = round(
            len(self.agents)
            * float(mig_inc_introduction_data.loc[self.current_year])
            / 1000
        )

        self.introduce_adult(number_of_BIRTH)

        if len(self.graph.nodes()) > 0:
            self.next_node_id = max(
                self.next_node_id,
                max(self.graph.nodes()) + 1
            )

        # Collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        """Run model for desired number of iterations."""

        for _ in range(n):
            self.step()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def get_pct_data(self):
        """Return percentage data by state."""

        data = self.datacollector.get_model_vars_dataframe()

        pct_data = data[
            [
                "% NEVERSMOKER",
                "% SMOKER",
                "% QUITTER"
                # "% DEAD"
            ]
        ]

        pct_data.index = range(
            self.year_to_start,
            1 + self.year_to_end
        )

        return pct_data.round(2)

    def print_pct_data(self):
        """Print percentage data by state."""

        print(self.get_pct_data())

    def get_network_nodes_dataframe(self):
        """Return current node-level network information."""

        return pd.DataFrame({
            "node": list(self.graph.nodes()),
            "age": [
                self.graph.nodes[node].get("age")
                for node in self.graph.nodes()
            ],
            "status": [
                self.graph.nodes[node].get("status")
                for node in self.graph.nodes()
            ],
            "status_code": [
                self.graph.nodes[node].get("status_code")
                for node in self.graph.nodes()
            ],
            "degree": [
                self.graph.degree[node]
                for node in self.graph.nodes()
            ],
            "initial_age": [
                self.graph.nodes[node].get("initial_age")
                for node in self.graph.nodes()
            ],
            "initial_degree": [
                self.graph.nodes[node].get("initial_degree")
                for node in self.graph.nodes()
            ],
            "agent_id": [
                self.graph.nodes[node].get("agent_id")
                for node in self.graph.nodes()
            ]
        })

    def get_network_edges_dataframe(self):
        """Return current edge-level network information."""

        return pd.DataFrame([
            {
                "node_1": u,
                "node_2": v,
                "age_1": self.graph.nodes[u].get("age"),
                "age_2": self.graph.nodes[v].get("age"),
                "status_1": self.graph.nodes[u].get("status"),
                "status_2": self.graph.nodes[v].get("status"),
                "degree_1": self.graph.degree[u],
                "degree_2": self.graph.degree[v]
            }
            for u, v in self.graph.edges()
        ])

#%% Agent

class NSQ_Agent(mesa.Agent):
    """Define the agent-level actions as NSQ_Agent."""

    def __init__(
            self,
            model,
            initial_age,
            initial_state
            ):
        """Initialise the NSQ_Agent class."""

        super().__init__(model)

        self.current_age = initial_age

        self.current_state = initial_state
        self.updated_state = initial_state

    def NEVERSMOKER_initiation(self):
        """Smoking initiation for NEVERSMOKER."""

        neighbours = self.model.grid.get_neighbors(
            self.pos,
            include_center=False
        )

        SMOKER_neighbours = list(
            filter(
                lambda agent: agent.current_state is State.SMOKER,
                neighbours
            )
        )

        if self.random.random() < self.model.delta_N_to_S:
            self.updated_state = State.SMOKER

        elif len(neighbours) != 0:

            prob_N_to_S_due_to_S = (
                len(SMOKER_neighbours)
                / len(neighbours)
            ) * (
                1
                - (
                    1 - self.model.beta_N_to_S_due_to_S_est
                ) ** len(SMOKER_neighbours)
            )

            if self.random.random() < prob_N_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state

        else:
            self.updated_state = self.current_state

    def SMOKER_cessation(self):
        """Smoking cessation for SMOKER."""

        neighbours = self.model.grid.get_neighbors(
            self.pos,
            include_center=False
        )

        NEVERSMOKER_neighbours = list(
            filter(
                lambda agent: agent.current_state is State.NEVERSMOKER,
                neighbours
            )
        )

        QUITTER_neighbours = list(
            filter(
                lambda agent: agent.current_state is State.QUITTER,
                neighbours
            )
        )

        if self.random.random() < self.model.delta_S_to_Q:
            self.updated_state = State.QUITTER

        elif len(neighbours) != 0:

            prob_S_to_Q_due_to_N = (
                len(NEVERSMOKER_neighbours)
                / len(neighbours)
            ) * (
                1
                - (
                    1 - self.model.beta_S_to_Q_due_to_N
                ) ** len(NEVERSMOKER_neighbours)
            )

            prob_S_to_Q_due_to_Q = (
                len(QUITTER_neighbours)
                / len(neighbours)
            ) * (
                1
                - (
                    1 - self.model.beta_S_to_Q_due_to_Q_est
                ) ** len(QUITTER_neighbours)
            )

            if self.random.random() < prob_S_to_Q_due_to_N:
                self.updated_state = State.QUITTER

            elif self.random.random() < prob_S_to_Q_due_to_Q:
                self.updated_state = State.QUITTER

            else:
                self.updated_state = self.current_state

        else:
            self.updated_state = self.current_state

    def QUITTER_relapse(self):
        """Smoking relapse for QUITTER."""

        neighbours = self.model.grid.get_neighbors(
            self.pos,
            include_center=False
        )

        SMOKER_neighbours = list(
            filter(
                lambda agent: agent.current_state is State.SMOKER,
                neighbours
            )
        )

        if self.random.random() < self.model.delta_Q_to_S:
            self.updated_state = State.SMOKER

        elif len(neighbours) != 0:

            prob_Q_to_S_due_to_S = (
                len(SMOKER_neighbours)
                / len(neighbours)
            ) * (
                1
                - (
                    1 - self.model.beta_Q_to_S_due_to_S
                ) ** len(SMOKER_neighbours)
            )

            if self.random.random() < prob_Q_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state

        else:
            self.updated_state = self.current_state

    def step(self):
        """Decide updated state for one iteration."""

        if self.current_state is State.NEVERSMOKER:
            self.NEVERSMOKER_initiation()

        elif self.current_state is State.SMOKER:
            self.SMOKER_cessation()

        elif self.current_state is State.QUITTER:
            self.QUITTER_relapse()

    def advance(self):
        """Apply state update, age agent, and sync graph node information."""

        self.current_age += 1
        self.current_state = self.updated_state

        if self.pos is not None and self.pos in self.model.graph.nodes:
            self.model.graph.nodes[self.pos]["age"] = int(self.current_age)
            self.model.graph.nodes[self.pos]["status"] = self.current_state.name
            self.model.graph.nodes[self.pos]["status_code"] = self.current_state.value
            self.model.graph.nodes[self.pos]["agent_id"] = self.unique_id

#%%

best_graph_for_abm = best_network_info["G"]
best_node_age_for_abm = best_network_info["node_age"]

#%%

def simulator(
              year_to_start, year_to_end,
              delta_N_to_S, delta_S_to_Q, delta_Q_to_S,
              beta_SN_to_QN, beta_QS_to_SS,
              # timesteps, # 1974-2023 (excluding the initial settings)
              batch_size = 1, random_state = None
              ):
    """"Define the NSQ_Model simulator on the synthetic network"""
    model = NSQ_Model_UK(
        year_to_start = year_to_start,
        year_to_end = year_to_end,
        initial_NEVERSMOKER_pct = interpolated_data.loc[year_to_start, '% NEVERSMOKER'] / 100,
        initial_SMOKER_pct = interpolated_data.loc[year_to_start, '% SMOKER'] / 100,
        initial_QUITTER_pct = interpolated_data.loc[year_to_start, '% QUITTER'] / 100,
        delta_N_to_S = delta_N_to_S,
        delta_S_to_Q = delta_S_to_Q,
        delta_Q_to_S = delta_Q_to_S,
        beta_S_to_Q_due_to_N = beta_SN_to_QN,
        beta_Q_to_S_due_to_S = beta_QS_to_SS,
        input_graph=best_graph_for_abm,
        input_node_age=best_node_age_for_abm,
        )
    
    model.run_model(year_to_end - year_to_start)

    # return model    
    return model.get_pct_data()

#%%

a = simulator(
    1974,
    2023,
    0.01,   # delta_N_to_S
    0.04,   # delta_S_to_Q: 0.03-0.05
    0.10,   # delta_Q_to_S: 0.1-0.15
    0.20,   # beta_S_to_Q_due_to_N
    0.01    # beta_Q_to_S_due_to_S
)

# aa = list(agent.current_age for agent in a.agents)

#%%

# ----------

#%%

class State(Enum):
    """Define the possible smoking states for agents:"""
    
    NEVERSMOKER = 0
    SMOKER = 1
    QUITTER = 2
    DEAD = 3
    

#%% Agent-Based Model:

class NSQ_Model_UK(mesa.Model):
    """Define the model-level actions as NSQ_Model:"""
    
    def __init__(
            self,
            # Set seed (if necessary):
            # seed = ,
            # Year to start/end
            year_to_start, # 1974
            year_to_end, # 2023
            # Initial population parameters (%, NEVERSMOKER, SMOKER, QUITTER):
            initial_NEVERSMOKER_pct, # 37.4% in 1974
            initial_SMOKER_pct, # 45.6% in 1974
            initial_QUITTER_pct, # 17.0% in 1974
            # Parameters to be calibrated (5):
            delta_N_to_S,
            delta_S_to_Q,
            delta_Q_to_S,
            beta_S_to_Q_due_to_N,
            beta_Q_to_S_due_to_S,
            # Embedded network's structure: (not specified now, specify when implementing)
            network_type,
            # Calibrated parameters (2):
            beta_N_to_S_due_to_S = 0.40719,
            beta_S_to_Q_due_to_Q = 0.35214,
            # Embedded network's details: (not specified now, specify when implementing)
            **network_parameters
            ):
        """Initialise the NSQ_Model class:"""
        
        # Initialise Mesa's 'Model' class:
        super().__init__() # super().__init__(seed) if seed != None
        
        # Pass the values of spontaneous parameters to the model class which is under construction:
        self.delta_N_to_S = delta_N_to_S
        self.delta_S_to_Q = delta_S_to_Q
        self.delta_Q_to_S = delta_Q_to_S
        
        # Pass the values of interaction parameters to the model class which is under construction:
        self.beta_N_to_S_due_to_S = beta_N_to_S_due_to_S
        self.beta_S_to_Q_due_to_N = beta_S_to_Q_due_to_N
        self.beta_S_to_Q_due_to_Q = beta_S_to_Q_due_to_Q
        self.beta_Q_to_S_due_to_S = beta_Q_to_S_due_to_S
        
        # Define estimated interaction parameters (from Christakis):
        self.beta_N_to_S_due_to_S_est = 1 - (1 - beta_N_to_S_due_to_S * (1 - (1 - delta_N_to_S)**4.5)) ** (1/4.5)
        self.beta_S_to_Q_due_to_Q_est = 1 - (1 - beta_S_to_Q_due_to_Q * (1 - (1 - delta_S_to_Q)**4.5)) ** (1/4.5)
        
        # Pass the information of year to start/end to the model class which is under construction:
        self.year_to_start = year_to_start
        self.year_to_end = year_to_end
        self.current_year = year_to_start
        self.step_count = 0
        
        # Pass the values of initial population parameters to the model class which is under construction:
        self.initial_NEVERSMOKER_pct = initial_NEVERSMOKER_pct
        self.initial_SMOKER_pct = initial_SMOKER_pct
        self.initial_QUITTER_pct = initial_QUITTER_pct
        self.initial_DEAD_pct = 1 - (initial_NEVERSMOKER_pct + initial_SMOKER_pct + initial_QUITTER_pct)
        
        # Pass the embedded network's details to the model class which is under construction:
        self.graph = network_type(**network_parameters)
        # self.graph = network_type(*network_parameters.values()) "when wrapping network parameters as a dictionary"
        self.grid = mesa.space.NetworkGrid(self.graph)
        
        #
        self.next_node_id = max(self.graph.nodes) + 1
        
        # print(f"Graph has {len(self.graph.nodes())} nodes", flush = True) "testing the # of nodes loaded"
        
        # Adding agents to nodes:
        for node in self.graph.nodes(): # [!] self.graph.nodes(): an iterable NodeView, not a list
            assigned_age = self.assign_initial_age()
            agent = NSQ_Agent(self, assigned_age, State.NEVERSMOKER) # Mesa 3 migration: no need to pass unique_id explicitly
            # self.schedule.add(agent) # Mesa 3 migration: eliminating the need to manually call ~"
            self.grid.place_agent(agent, node) # [!] Mesa automatically assigns agent.pos = node
            # when calling agent class, the objects in agent class are automatically passed
            # (agent, node) = (i+1, i), i starts from 0
            # [!] implicitly guarantee # of agents and # of nodes are equal
        
        # Randomly choose initial SMOKER from all agents (choose nodes first):
        initial_SMOKER_nodes = self.random.sample(list(self.graph.nodes()), round(self.initial_SMOKER_pct * len(list(self.graph.nodes()))))
        initial_SMOKER_agents = self.grid.get_cell_list_contents(initial_SMOKER_nodes)
        for agent in initial_SMOKER_agents:
            agent.current_state = State.SMOKER
            agent.updated_state = State.SMOKER
        
        # Randomly choose initial QUITTER from all remaining agents (choose nodes first):
        initial_QUITTER_nodes = self.random.sample(
            list(filter(lambda node: node not in initial_SMOKER_nodes, list(self.graph.nodes()))),
            round(self.initial_QUITTER_pct * len(list(self.graph.nodes())))
            )
        initial_QUITTER_agents = self.grid.get_cell_list_contents(initial_QUITTER_nodes)
        for agent in initial_QUITTER_agents:
            agent.current_state = State.QUITTER
            agent.updated_state = State.QUITTER
        
        # Randomly choose initial DEAD from all remaining agents (choose nodes first):
        initial_DEAD_nodes = self.random.sample(
            list(filter(lambda node: node not in initial_SMOKER_nodes and node not in initial_QUITTER_nodes, list(self.graph.nodes()))),
            round(self.initial_DEAD_pct * len(list(self.graph.nodes())))
            )
        initial_DEAD_agents = self.grid.get_cell_list_contents(initial_DEAD_nodes)
        for agent in initial_DEAD_agents:
            agent.current_state = State.DEAD
            agent.updated_state = State.DEAD
        
        # Generate initial NEVERSMOKER by deleting initial SMOKER and initial QUITTER:
        initial_NEVERSMOKER_nodes = list(filter(lambda node: node not in initial_SMOKER_nodes and node not in initial_QUITTER_nodes and node not in initial_DEAD_nodes, list(self.graph.nodes())))
        initial_NEVERSMOKER_agents = self.grid.get_cell_list_contents(initial_NEVERSMOKER_nodes)
        for agent in initial_NEVERSMOKER_agents:
            agent.current_state = State.NEVERSMOKER
            agent.updated_state = State.NEVERSMOKER
        
        # Add DataCollector to track desired data:
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # Agents' list in each state: [!] collecting this causes the execution to be extremely slow
                # "NEVERSMOKER_agents": lambda m: m.get_agents_by_state(State.NEVERSMOKER),
                # "SMOKER_agents": lambda m: m.get_agents_by_state(State.SMOKER),
                # "QUITTER_agents": lambda m: m.get_agents_by_state(State.QUITTER),
                # "DEAD_agents": lambda m: m.get_agents_by_state(State.DEAD),
                
                # Node lists of agents in each state:
                "NEVERSMOKER_nodes": lambda m: m.get_nodes_by_state(State.NEVERSMOKER),
                "SMOKER_nodes": lambda m: m.get_nodes_by_state(State.SMOKER),
                "QUITTER_nodes": lambda m: m.get_nodes_by_state(State.QUITTER),
                "DEAD_nodes": lambda m: m.get_nodes_by_state(State.DEAD),
        
                # Counts of agents in each state:
                "NEVERSMOKER_count": lambda m: m.count_agents_by_state(State.NEVERSMOKER),
                "SMOKER_count": lambda m: m.count_agents_by_state(State.SMOKER),
                "QUITTER_count": lambda m: m.count_agents_by_state(State.QUITTER),
                "DEAD_count": lambda m: m.count_agents_by_state(State.DEAD),
                
                # Percentages of agents in each state:
                "% NEVERSMOKER": lambda m: m.pct_agents_by_state(State.NEVERSMOKER),
                "% SMOKER": lambda m: m.pct_agents_by_state(State.SMOKER),
                "% QUITTER": lambda m: m.pct_agents_by_state(State.QUITTER),
                "% DEAD": lambda m: m.pct_agents_by_state(State.DEAD)
            },
            agent_reporters={
                "Age": lambda a: a.current_age, # 
                "State": lambda a: a.current_state, # "current_state.name"
                "Agent": lambda a: a.unique_id, # 
                "Node": lambda a: a.pos, # "pos"
            }
        )
        
        # Capture desired data from initial settings (step 0):
        self.datacollector.collect(self)
    
    def assign_initial_age(self):
        """"""
        probabilities = population_by_age_data.loc[self.year_to_start]
        ages = probabilities.index.tolist()
        weights = probabilities.values.tolist()
        
        assigned_age = self.random.choices(
                           population = ages,
                           weights = weights,
                           k = 1)[0]
        
        if assigned_age == "100+":
            return self.random.randint(100, 120) # oldest people in Great Britain is about age 120.
        else:
            return int(assigned_age)
    
    def get_agents_by_state(self, state):
        """Return list of agents in a given state:"""
        return list(filter(lambda agent: agent.current_state is state, self.agents))
    
    def get_nodes_by_state(self, state):
        """Return list of nodes where their agents in a given state:"""
        return list(map(lambda agent: agent.pos, self.get_agents_by_state(state)))
    
    def count_agents_by_state(self, state):
        """Calculate counts of agents in a given state:"""
        return len(self.get_agents_by_state(state))

    def pct_agents_by_state(self, state):
        """Calculate percentage of agents in a given state:"""
        return (self.count_agents_by_state(state) / len(self.agents)) * 100
    
    #######
    
    def mortality_probability(self, agent):
        """"""
        # Ages 0, 1, ..., 99, 100+
        if agent.current_age >= 100:
            age_column = "100+"
        else:
            age_column = str(agent.current_age)

        # 
        if agent.current_state is State.NEVERSMOKER:
            mortality_probability = float(mortality_by_age_data_NEVERSMOKER.loc[self.current_year, age_column])
            
        elif agent.current_state is State.SMOKER:
            mortality_probability = float(mortality_by_age_data_SMOKER.loc[self.current_year, age_column])
            
        elif agent.current_state is State.QUITTER:
            mortality_probability = float(mortality_by_age_data_QUITTER.loc[self.current_year, age_column])
        
        return min(mortality_probability, 1.0)
    
    def remove_dead(self): # by age and smoking status
        """"""
        dead_agents_to_remove = []
        
        for agent in list(self.agents):
            
            if agent.current_state is State.DEAD:
                continue

            death_probability = self.mortality_probability(agent)

            if self.random.random() < death_probability:
                dead_agents_to_remove.append(agent)
                
        dead_nodes_to_remove = list(agent.pos for agent in dead_agents_to_remove)
        
        for agent in dead_agents_to_remove:
            agent.current_state = State.DEAD
            agent.updated_state = State.DEAD

            self.grid.remove_agent(agent)
            agent.remove()

        self.graph.remove_nodes_from(dead_nodes_to_remove)
        
        return self.agents, self.graph
    
    #######
    
    def get_agent_at_node(self, node):
        """Return the single agent located at a node."""
        agents = self.grid.get_cell_list_contents([node])

        if len(agents) == 0:
            return None

        return agents[0]
    
    def age_fitness(self, age, min_age = 16, age_scale = 20):
        """
        Fitness decreases with age:
            
            e^("-"age_difference/age_scale)

        age_scale controls how quickly fitness declines.
        Larger age_scale = slower decline.
        Smaller age_scale = faster decline.
        """
        
        age_difference = max(0, age - min_age)

        return math.exp(- age_difference / age_scale)
    
    def node_attractiveness(self, node, degree_exponent = 1.0, age_scale = 20):
        """
        Bianconi-Barabási-style attractiveness:

            attractiveness = fitness(age) * degree^degree_exponent

        degree_exponent = 1.0 gives standard preferential attachment.
        degree_exponent < 1.0 weakens degree advantage.
        degree_exponent > 1.0 strengthens degree advantage.
        """

        agent = self.get_agent_at_node(node)

        if agent is None:
            return 0.0

        if agent.current_state is State.DEAD:
            return 0.0

        degree = self.graph.degree(node)

        # Add 1 so zero-degree nodes still have a chance
        degree_component = (degree + 1) ** degree_exponent

        fitness_component = self.age_fitness(
                                             agent.current_age,
                                             min_age = 16,
                                             age_scale = age_scale
                                            )

        return fitness_component * degree_component
    
    # This means a highly connected young agent is very attractive, while an old agent becomes less attractive even if they have a high degree.
    
    def weighted_sample_without_replacement(self, items, weights, k):
        """Sample k items without replacement using weights."""
        selected = []

        items = list(items)
        weights = list(weights)

        k = min(k, len(items))

        for _ in range(k):
            total_weight = sum(weights)

            if total_weight <= 0:
                chosen_index = self.random.randrange(len(items))
            else:
                r = self.random.random() * total_weight
                cumulative = 0

                chosen_index = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if r <= cumulative:
                        chosen_index = i
                        break

            selected.append(items.pop(chosen_index))
            weights.pop(chosen_index)

        return selected
    
    def introduce_adult(
                        self,
                        number_of_BIRTH,
                        new_agent_age = 16,
                        average_degree = 3,
                        age_scale = 20,
                        degree_exponent = 1.0
                        ):
        """
        Introduce new agents using age-fitness preferential attachment.

        New agents first enter the system, then each new agent creates average_degree edges.

        Target probability is proportional to:

            attractiveness = fitness(age) * (degree + 1)^degree_exponent

        where fitness(age) decreases with age.
        """
        
        new_nodes = []

        # --------------------------------------------------
        # Step 1: add all new nodes and agents first
        # --------------------------------------------------
        for _ in range(number_of_BIRTH):

            new_node = self.next_node_id
            self.next_node_id += 1

            self.graph.add_node(new_node)
            self.graph.nodes[new_node]["agent"] = []

            agent = NSQ_Agent(
                              self,
                              new_agent_age,
                              State.NEVERSMOKER
                              )

            self.grid.place_agent(agent, new_node)

            new_nodes.append(new_node)

        # --------------------------------------------------
        # Step 2: connect each newly added node
        # --------------------------------------------------
        for new_node in new_nodes:

            candidate_nodes = [
                               node for node in self.graph.nodes
                               if node != new_node
                               ]

            weights = [
                       self.node_attractiveness(
                                                node,
                                                degree_exponent = degree_exponent,
                                                age_scale = age_scale
                                                )
                       for node in candidate_nodes
                       ]

            number_of_connections = min(average_degree, len(candidate_nodes))

            if number_of_connections <= 0:
                continue

            targets = self.weighted_sample_without_replacement(candidate_nodes, weights, number_of_connections)

            for target in targets:
                self.graph.add_edge(new_node, target)

        return self.agents, self.graph
    
    #######
     
    def step(self): #
        """Advance model by one step:"""
        
        self.current_year += 1
        
        # Existing agents update smoking status and age
        self.agents.do("step")
        self.agents.do("advance")
        
        # Remove agents by age/status mortality rule
        self.remove_dead()
        self.next_node_id = max(self.graph.nodes) + 1
        
        # Introduce new adult agents using fitness/attractiveness rule
        self.introduce_adult(round(len(self.agents) * float(crude_introduction_data.loc[self.current_year]) / 1000))
        self.next_node_id = max(self.graph.nodes) + 1
        
        # Collect data
        self.datacollector.collect(self)
        
    def run_model(self, n):
        """Run model for desired number of iterations"""
        for _ in range(n):
            self.step()
         
    def get_pct_data(self):
        """Returns percentage data by state:"""
        data = self.datacollector.get_model_vars_dataframe()
        pct_data = data[["% NEVERSMOKER", "% SMOKER", "% QUITTER"
                         # "% DEAD"
                         ]]
        pct_data.index = range(self.year_to_start, 1 + self.year_to_end) # index starts from where simulation initiated
        # years_to_omit = list(range(1975, 2000, 2)) # [1975, 1977, ..., 1999]
        # pct_data = pct_data.drop(index = years_to_omit)
        return pct_data.round(2)
     
    def print_pct_data(self):
        """Print percentage data by state:"""
        print(self.get_pct_data())

#%% Agent:

class NSQ_Agent(mesa.Agent):
    """Define the agent-level actions as NSQ_Agent:"""
    
    def __init__(
            self,
            model,
            initial_age, # initially assigned age
            initial_state # initially assigned smoking state
            ):
        """Initialise the NSQ_Agent class:"""
        
        # Initialise Mesa's 'Agent' class and pass objects from the 'model' class to the agent class which is under construction:
        super().__init__(model)
        
        # Define current age before 'step()' and initialise:
        self.current_age = initial_age

        # Define current state before state change and initialise:
        self.current_state = initial_state
        
        # Define updated state after state change and initialise:
        self.updated_state = initial_state
        
    def NEVERSMOKER_initiation(self):
        """Define the dynamics of smoking initiation for NEVERSMOKER:"""
        
        # Identify neighbouring agents and those who are SMOKER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_N_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            # prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S)**len(SMOKER_neighbours))
            prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S_est)**len(SMOKER_neighbours))
            if self.random.random() < prob_N_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours
    
    def SMOKER_cessation(self):
        """Define the dynamics of smoking cessation for SMOKER:"""
        
        # Identify neighbouring agents and those who are NEVERSMOKER and QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        NEVERSMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.NEVERSMOKER, neighbours))
        # NEVERSMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.NEVERSMOKER)]
        QUITTER_neighbours = list(filter(lambda agent: agent.current_state is State.QUITTER, neighbours))
        # QUITTER_neighbours = [agent for agent in neighbours if (agent.current_state is State.QUITTER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_S_to_Q:
            self.updated_state = State.QUITTER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_S_to_Q_due_to_N = (len(NEVERSMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_N)**len(NEVERSMOKER_neighbours))
            # prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q)**len(QUITTER_neighbours))
            prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q_est)**len(QUITTER_neighbours))
            if self.random.random() < prob_S_to_Q_due_to_N:
                self.updated_state = State.QUITTER
            elif self.random.random() < prob_S_to_Q_due_to_Q:
                self.updated_state = State.QUITTER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, NEVERSMOKER_neighbours, QUITTER_neighbours
    
    def QUITTER_relapse(self):
        """Define the dynamics of smoking relapse for QUITTER:"""
        
        # Identify neighbouring agents and those who are QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_Q_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_Q_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_Q_to_S_due_to_S)**len(SMOKER_neighbours))
            if self.random.random() < prob_Q_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours

    def step(self):
        """Decide what updated state should be for one iteration:"""
        
        # Smoking initiation for NEVERSMOKER:
        if self.current_state is State.NEVERSMOKER:
            self.NEVERSMOKER_initiation()
        
        # Smoking cessation for SMOKER:
        elif self.current_state is State.SMOKER:
            self.SMOKER_cessation()
        
        # Smoking relapse for QUITTER:
        elif self.current_state is State.QUITTER:
            self.QUITTER_relapse()
    
    def advance(self):
        """Apply any state change decided in step() to the agent's current state:"""
        self.current_age += 1 # 
        self.current_state = self.updated_state
    
#%% Simulator on Erdös-Rényi Graph

def simulator_ER_UK(
                    year_to_start, year_to_end,
                    delta_N_to_S_ER_UK, delta_S_to_Q_ER_UK, delta_Q_to_S_ER_UK,
                    beta_SN_to_QN_ER_UK, beta_QS_to_SS_ER_UK,
                    # timesteps, # 1974-2023 (excluding the initial settings)
                    batch_size = 1, random_state = None
                    ):
    """"Define the NSQ_Model simulator on Erdös-Rényi graph"""
    model_ER_UK = NSQ_Model_UK(
        year_to_start = year_to_start,
        year_to_end = year_to_end,
        initial_NEVERSMOKER_pct = interpolated_data.loc[year_to_start, '% NEVERSMOKER'] / 100,
        initial_SMOKER_pct = interpolated_data.loc[year_to_start, '% SMOKER'] / 100,
        initial_QUITTER_pct = interpolated_data.loc[year_to_start, '% QUITTER'] / 100,
        delta_N_to_S = delta_N_to_S_ER_UK,
        delta_S_to_Q = delta_S_to_Q_ER_UK,
        delta_Q_to_S = delta_Q_to_S_ER_UK,
        beta_S_to_Q_due_to_N = beta_SN_to_QN_ER_UK,
        beta_Q_to_S_due_to_S = beta_QS_to_SS_ER_UK,
        network_type = nx.erdos_renyi_graph,
        # network_parameters
        n = 10000,
        p = 0.004
        )
    
    model_ER_UK.run_model(year_to_end - year_to_start)

    # return model_ER_UK    
    return model_ER_UK.get_pct_data()

#%%

b = simulator_ER_UK(
    1974,
    2023,
    0.01,
    0.04,
    0.10,
    0.20,
    0.01
)

# bb = list(agent.current_age for agent in b.agents)

#%%

from matplotlib.ticker import MultipleLocator

plt.figure(figsize = (9, 6), dpi = 1000)
plt.scatter(np.linspace(1974, 2023, 50), b['% NEVERSMOKER'], label = 'New Model', color = 'lightcoral', marker = 'v')
plt.scatter(np.linspace(1974, 2023, 50), a['% NEVERSMOKER'], label = 'Old Model', color = 'lightseagreen', marker = '^')
plt.scatter(np.linspace(1974, 2023, 50), interpolated_data['% NEVERSMOKER'], label = 'ONS Data', color = 'black', marker = 'o')
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.xlabel("Year")
plt.ylabel("% NEVERSMOKER")
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5, alpha = 0.7, color = 'gray')
plt.legend()
plt.show()

plt.figure(figsize = (9, 6), dpi = 1000)
plt.scatter(np.linspace(1974, 2023, 50), b['% SMOKER'], label = 'New Model', color = 'lightcoral', marker = 'v')
plt.scatter(np.linspace(1974, 2023, 50), a['% SMOKER'], label = 'Old Model', color = 'lightseagreen', marker = '^')
plt.scatter(np.linspace(1974, 2023, 50), interpolated_data['% SMOKER'], label = 'ONS Data', color = 'black', marker = 'o')
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.xlabel("Year")
plt.ylabel("% SMOKER")
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5, alpha = 0.7, color = 'gray')
plt.legend()
plt.show()

plt.figure(figsize = (9, 6), dpi = 1000)
plt.scatter(np.linspace(1974, 2023, 50), b['% QUITTER'], label = 'New Model', color = 'lightcoral', marker = 'v')
plt.scatter(np.linspace(1974, 2023, 50), a['% QUITTER'], label = 'Old Model', color = 'lightseagreen', marker = '^')
plt.scatter(np.linspace(1974, 2023, 50), interpolated_data['% QUITTER'], label = 'ONS Data', color = 'black', marker = 'o')
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.xlabel("Year")
plt.ylabel("% QUITTER")
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5, alpha = 0.7, color = 'gray')
plt.legend()
plt.show()

#%%

from collections import Counter

plt.figure(figsize = (9, 6), dpi = 1000)
proportions_aa = {
               k: v / len(aa)
               for k, v in Counter(aa).items()
               }

plt.bar(
        proportions_aa.keys(),
        proportions_aa.values(),
        label = 'Age Distribution of New Model for 2023; N = 10454',
        color = 'lightseagreen'
        )

plt.scatter(np.linspace(16, 101, 85), population_by_age_data.loc[2023],
            label = 'ONS Data for 2023', color = 'black')

plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure(figsize = (9, 6), dpi = 1000)
proportions_bb = {
               k: v / len(bb)
               for k, v in Counter(bb).items()
               }

plt.bar(
        proportions_bb.keys(),
        proportions_bb.values(),
        label = 'Age Distribution of Old Model for 2023; N = 9344',
        color = 'lightcoral'
        )

plt.scatter(np.linspace(16, 101, 85), population_by_age_data.loc[2023],
            label = 'ONS Data for 2023', color = 'black')

plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show()

#%%





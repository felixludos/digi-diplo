# Digital Diplomat

This repo contains a suite of python scripts for creating, managing, and visualizing [Diplomacy](https://en.wikipedia.org/wiki/Diplomacy_(game)) games. The scripts herein are designed to bridge the gap between the mechanics of the game (the adjudicator used here is [`pydip`](https://github.com/aparkins/pydip)) and a user facing interface to play the game.

The primary use case here is for groups that play Diplomacy over the internet managed by a moderator. After the players submit their moves, the moderator can run these scripts (specifically `diplo-step` and `render`) to automatically update the game state and visualize the updated game state to send to the players. Furthermore, the moderator can use other scripts (`mapping`, `collect-pos`, `collect-fills`) to play the game with custom maps.

## Installation

Firstly, you must have `python` and `pip` to run the scripts and install all dependencies (it's recommended you do so using the [Anaconda distribution](https://www.anaconda.com/products/individual)).

1. Clone this repo in the directory of your choice
    
    ```bash
    git clone https://github.com/felixludos/digi-diplo
    cd digi-diplo
    ```

2. From inside the repo, install all dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Check to make sure all the scripts are available. This command should list all the scripts discussed below together with a short description.

    ```bash
    fig -h
   ```


## Scripts

All of these scripts require and/or create [yaml](https://yaml.org/) data files.

### `node-locs`

Collects a list of coordinates on an image for each node (usually called in `mapping`).

#### Usage

Left-click - select coordinates for the named node
Right-click - return to the previous node
All selected coordinates are saved automatically to the provided nodes yaml file when the window is closed.

#### Example Commands

    fig node-locs --nodes-path test/1/nodes.yaml --image-path assets/classic.gif
    fig node-locs test/node-locs


### `mapping`

Given a list of nodes, this script first uses the `node-locs` script to get a position for each node and then a list of all edges between nodes.

#### Usage

Once all the node locations are specified, you can specify the edges between nodes first for armies then fleets.

First, a window will open for the edges of armies.
Click on a node location to add an edge between 
Press space to move to the next node
Press backspace to clear all selected edges and move to the previous node.
All selected coordinates are saved automatically to the an edges yaml file when the window is closed.
After the army edges are specified, another window opens for the fleets.

#### Example Commands

    fig mapping --nodes-path test/1/nodes.yaml --edges-path test_edges.yaml --image-path assets/classic.gif
    fig mapping test/mapping
    

### `collect-pos`

Collects the coordinates on an image for units, texts, and centers used when rendering game states.

#### Usage

Left-click - select coordinates for the named node
Right-click - clear all specified coordinates for the current node and return to the previous node
All selected coordinates are saved automatically to the an pos yaml file when the window is closed.

#### Example Commands

    fig collect-pos --nodes-path test/2/nodes.yaml --edges-path test/2/edges.yaml --image-path assets/classic.gif --out test_pos.yaml
    fig collect-pos test/collect-pos


### `collect-fills`

Collects the coordinates on an image to flood fill with colors in rendering

#### Usage

Click on every location that should be flood-filled when rendering a game state based on the owner of the territory.
Press space to move to the next territory
All selected coordinates are saved automatically to the pos yaml file when the window is closed.

#### Example Commands

    fig collect-fills --nodes-path test/3/nodes.yaml --pos-path test/3/pos.yaml --image-path assets/classic.gif
    fig collect-fills test/collect-fills


### `diplo-new`

Given the map and initial player info, this creates an initial game state yaml file.

#### Usage

This script requires the nodes and edges yaml files, as well as the players file which contains the starting ownership and units for every player.

#### Example Commands

    fig diplo-new --root test/4 --save-path test_state.yaml
    fig diplo-new test/diplo-new

### `diplo-step` 

Given the current game state (as a yaml file) and the selected actions for all players (also a yaml file), this updates the games state using the `pydip` adjudicator.

#### Usage

This script requires the current state and actions (see `test/5` for examples) yaml files, provided as `state-path` and `action-path`. The output state should be saved at `save-path`.

#### Example Commands

    fig diplo-step --root test/4 --state-path test/5/state.yaml --action-path test/5/actions.yaml --save-path test_new_state.yaml
    fig diplo-step test/diplo-step
    

### `render` 

Given the map, an image of the map, current game state, and optionally current actions, this draws the game state on the image (using the coordinates collected by `collect-pos` and `collect-fills`) 

#### Usage

You must include a yaml file path containing the game state as `state-path`, and optionally a path to the next actions as `action-path`. To save the rendered state pass in a path `save-path`.

If you want to customize the way the game state is customized (sizes/shapes of markers or arrows), it is strongly recommended that a config file is created similar to `viz/classic` (located at `config/viz/classic.yaml`).

#### Example Commands

    fig render viz/classic --state-path test/5/state.yaml
    fig render viz/classic --state-path test/5/state.yaml --action-path test/5/actions.yaml
    fig render test/render
    

### `parse-vdip`

From a text file of the full order log of a game on [web-diplomacy](http://webdiplomacy.net/) or [vdiplomacy](https://vdiplomacy.com/), this extracts the actions in the standard format for all seasons.

#### Usage

Copy and paste the full order log from any game on [web-diplomacy](http://webdiplomacy.net/) or [vdiplomacy](https://vdiplomacy.com/) into a text file. The path to this text file must then be passed in to this script as the `log-path`. You should also need to provide the path to the nodes (for the territory names) as `nodes-path`, and a directory to save all the output actions as `out-dir`.

#### Example Commands

    fig parse-vdip --log-path test/6/log.txt --nodes-path test/1/nodes.yaml --out-dir test_actions
    fig parse-vdip test/parse-vdip
    

### `diplo-traj`

Given a set of actions for all season (eg. parsed from a full order log), this script computes the game state after applying the actions for each season.

#### Usage

If you have a sequence of consecutive action files (eg. parsed from a web-diplomacy or vdiplomacy order log), then you can compute all the states with this script. As input the directory for all actions `action-dir` must be provided, and as output a path to a directory `state-dir` for all states must be provided. For the rules (including `nodes`, `edges`, and `players`), you can either provide a path to each file individually or a `root` path to a directory containing to all three.

#### Example Commands

    fig diplo-traj --root test/4 --state-dir test_states --action-dir test/7 --save-path test_new_state.yaml
    fig diplo-traj test/diplo-traj
    

### `render-traj`

Given a set of states and actions, this script visualizes every game state.

#### Usage

Visualizes the game state given a directory of states `state-dir` and actions `action-dir`. You can also provide a directory to save the rendered frames as `frame-dir`.

#### Example Commands

    fig render-traj viz/classic --root test/4 --state-dir test/8 --action-dir test/7 --frame-dir test_frames
    fig render-traj test/render-traj
    

## Configs

All config files (which are also yaml files) are in `config/`. The `config/test/` directory contains example configs for running several of the scripts. `config/viz` and `config/colors` contain config files used for rendering games states. 

## Create New Map

Since the Diplomacy adjudicator is independent of the specific map, you can create new custom maps with any number of territories (nodes) and players. Here's the high-level instructions to create a new map and run/visualize games with it.

1. Create a yaml file that lists all the territories (nodes) in the map. Each node must have an ID (a unique short name) and a type (land, coast, or sea). The node can also optionally have a full name, and it can designate the node as a supply center. If the node has multiple coasts those must also be listed. See `data/classic/nodes.yaml` for an example of the format for the official map.

2. Using the yaml file of the nodes, the `mapping` script provides a visual interface to select the neighbors of each node for both armies and fleets. This script will creates a yaml file with all the edges in the map (see `data/classic/edges.yaml` as an example).

3. Compile a yaml file of all players and starting units/ownership. See `data/classic/players.yaml` as an example.

4. Run the `collect-pos` script to specify the coordinates of where to draw the units and name for each node in the map (see `data/classic/pos.yaml`).

5. (Optional) Run the `collect-fills` script to specify the pixel locations of the regions that should be colored by ownership when visualizing a game state.

6. Start a new game by providing the map edges and nodes yaml files to the `diplo-new` script.

7. The game state yaml file can be visualized using the `render` script.

8. Once the actions are selected and specified in another yaml file (see examples in `data/`), you can resolve the actions and update the game state using the `diplo-step` script.


## TODO

- generate list of all possible actions for each player given the state and map (and script to select actions)
- add script to parse actions from raw text
- add progress bars to all scripts (especially `render`, `diplo-step`, and `diplo-traj`)

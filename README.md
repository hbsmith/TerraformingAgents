# TerraformingAgents

Package used to simulate life spreading throughout the galaxy. Built on Agents.jl.

## Setup

- Install Julia.

    > *Note*: It's helpful to add an alias to your `~/.zprofile` (or `.bashrc` if you're using bash) to access Julia from anywhere by typing `julia`. Add the following to that file: `alias julia="/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia" `

- Clone this repo (TerraformingAgents), or a forked repo to your machine

- Naviagate to your local copy of the repo. The rest of these instructions assume your pwd is `/Users/yourname/.../TerraformingAgents`
- Start the Julia REPL from your terminal (`julia`)


    ```julia
    julia> 
    ```
- Access Julia's package mode by hitting the `]` key. 

    ```julia
    julia> ]
    ```
- You will notice your promt change from `julia>` to `(@v1.6) pkg> `, or whatever your version of Julia is.

    ```julia
    (@v1.6) pkg> 
    ```
    > *Note*: This is Julia's powerful Pkg manager, you can learn more about it [here](https://docs.julialang.org/en/v1/stdlib/Pkg/).
    
    > *Note*: You can also access `shell` from within Julia by hitting `;` instead of `]`.

- Next we will activate our environment. This is kind of like a virtual environment within python--this allows for us to install packages at specific versions for this particular project without affecting other projects.

    ```julia
    (@v1.6) pkg> activate .
    ```
- You'll notice your prompt switch to the directory/repo name, to let you know you're in this environment. Then we `instantiate`. This means that Julia looks at the packages required by the project (inside the `Project.toml` file), and installs all of them.

    ```julia
    (TerraformingAgents) pkg> instantiate
    ```
- Next we will `activate` a subdirectory (specifically, `/scripts`). The reason we're separately activating this subdirectory is that it allows us to keep all the packages we require to run scripts separate from the requirements of using the package's source code.

    ```julia
    (TerraformingAgents) pkg> activate scripts/
    ```
- Just like when we activated the main directory, you'll notice the prompt changes to `(scripts)`. Next we have to install the `TerraformingAgents` package, since it's called by our script. But we can't do that in the usual way, since this is a local package and not in the global Julia registry. 
    
    > *Note*: You can install using `]add /Users/yourname/.../TerraformingAgents`. But instead of doing that, I would actually recommend installing using `]dev /Users/yourname/.../TerraformingAgents` instead. The difference is that changes to the TerraformingAgents source code from packages added using `dev` will be reflected whenever `TerraformingAgents` is imported. Otherwise, if we just use `add`, changes made to source code are only reflected when calling `update` on a package.

    ```julia
    (scripts) pkg> dev .
    ```
- Lastly, we have to `instantiate` to install those additional packages required to run our scripts.
    
    ```julia
    (scripts) pkg> instantiate
    ```
- Now, you should be ready to run the script `TA_data-interactive.jl`! From bash:

    ```bash
    julia scripts/TA_data-interactive.jl
    ```

- Or from the Julia REPL:
    ```julia
    include("scripts/TA_data-interactive.jl")
    ```

### Setup tl;dr

```julia
julia> ]activate .

(TerraformingAgents) pkg> instantiate

(TerraformingAgents) pkg> activate scripts/

(scripts) pkg> dev .

(scripts) pkg> instantiate
```

```bash
julia scripts/TA_data-interactive.jl
```

## Using the interactive Makie simulation

- If you adjust the nplanets slider, you must hit the update button, and then the step or run button for the number of planets in the simuation to reflect the update.

- To reset, double click the rest button (single click is insufficient)

- Hit run once to start running. Hit run again to stop running.

- Press enter to exit

- The green dots show Planet agents which have been visited by Life. The moving green dot shows the Life agent that's currently traveling to terraform a Planet. The red dot shows the Life agent's destination Planet agent.

- One plot shows the Mandel coefficient, which can be thought of at the correlation between distance and relatedness of planets. The other plot shows the p-value of this coefficient. 

> *Note*: Due to a few factors (slow calculation of the Mandel coefficient; overhead of the Makie interactive program), the interactive window may take a few minutes to appear. You can close the window, but to exit the process it is often necessary to hit `ENTER` in the running terminal.
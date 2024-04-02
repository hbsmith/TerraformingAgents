# TerraformingAgents

Package used to simulate life spreading throughout the galaxy. Built on Agents.jl.

Used for the simulations underlying the analyses in https://github.com/hbsmith/SmithSinapayen2024/ and Smith & Sinapayen, 2024 (https://arxiv.org/abs/2403.14195).

## Setup

- Install Julia (https://julialang.org/downloads/)

- Clone this repo (TerraformingAgents), or a forked repo to your machine

- Naviagate to your local copy of the repo. The rest of these instructions assume your pwd is `/Users/yourname/.../TerraformingAgents`
- Start the Julia REPL from your terminal (`julia`) 

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

- `add` the `TerraformingAgents` package via it's GitHub URL and verson number. 

    ```julia
    (@v1.6) pkg> add https://github.com/hbsmith/TerraformingAgents#0.1.0
    ```

### Setup tl;dr

```
% julia

julia> ]activate .

(@v1.6) pkg> add https://github.com/hbsmith/TerraformingAgents#0.1.0
```

module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot
using Distributions: Uniform
using NearestNeighbors

export Planet, Life, galaxy_model_basic, galaxy_model_advanced, galaxy_model_step!

"""Return random agent of type `A`"""
Agents.random_agent(
    model,
    A::Type{T},
    RNG::AbstractRNG = Random.default_rng(),
) where {T<:AbstractAgent} = model[rand(RNG, [k for (k, v) in model.agents if v isa A])]

magnitude(x::Tuple{<:Real,<:Real}) = sqrt(sum(x .^ 2))

"""Return normalized direction `start::AbstractAgent` to `finish::AbstractAgent`"""
direction(start::AbstractAgent, finish::AbstractAgent) =
    (finish.pos .- start.pos) ./ magnitude((finish.pos .- start.pos))

Base.@kwdef mutable struct Planet <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}

    composition::Vector{Int} ## Represents the planet's genotype
    initialcomposition::Vector{Int} ## Same as composition until it's terraformed
    alive::Bool
    claimed::Bool ## True if any Life has this planet as its destination

    ## Properties of the process, but not the planet itself
    ancestors::Vector{Planet} ## Planets that phylogenetically preceded this one
    parentplanet::Union{Planet,Nothing} ## Planet that directly preceded this one
    parentlife::Union{<:AbstractAgent,Nothing} #= This is of type Life, but I can't force
                                                  because it would cause a mutually 
                                                  recursive declaration b/w Planet and Life
                                               =#
    parentcomposition::Union{Vector{Int},Nothing}

end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    parentplanet::Planet
    composition::Vector{Int} ## Taken from parentplanet
    destination::Union{Nothing,Planet}
    ancestors::Vector{Life} ## Life agents that phylogenetically preceded this one
end

"""
Set up the galaxy (not user facing). 

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function galaxy_model_setup(detail::Symbol, kwarg_dict::Dict)

    @unpack RNG,
    extent,
    dt,
    interaction_radius,
    allowed_diff,
    lifespeed,
    compositionmaxvalue,
    compositionsize = kwarg_dict

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life},
        space2d,
        properties = @dict(dt, interaction_radius, allowed_diff, lifespeed),
    )

    if detail == :basic
        @unpack nplanets = kwarg_dict
        initialize_planets_basic!(
            nplanets,
            model;
            @dict(RNG, compositionmaxvalue, compositionsize)...,
        )
        ool = nothing
    elseif detail == :advanced
        @unpack pos, vel, planetcompositions, ool = kwarg_dict
        initialize_planets_advanced!(
            model;
            @dict(
                RNG,
                pos,
                vel,
                planetcompositions,
                compositionmaxvalue,
                compositionsize
            )...,
        )
    else
        throw(ArgumentError("`detail` must be `:basic` or `:advanced`"))
    end

    isnothing(ool) ? spawnlife!(random_agent(model, Planet, RNG), model) :
    spawnlife!(model.agents[ool], model)
    index!(model)
    model

end

"""
    galaxy_model_basic(nplanets; <keyword arguments>)

Create an Agents.jl `ABM` to simulate life spreading throughout the galaxy. Galaxy starts 
with `nplanets` number of planets.

...
# Arguments
- `RNG::AbstractRNG = Random.default_rng()`: RNG object
- `extent::Tuple{<:Real,<:Real} = (1, 1)`: Bounds of the agent space
- `dt::Real = 1.0`: Model timestep
- `interaction_radius::Union{Real,Nothing} = nothing`: How close `Life` and destination
        `Planet` have to be to interact via `interacting_pairs`. Default is `dt*lifespeed`.
- `allowed_diff::Real = 3`: How similar each element of a `Planet`'s `composition` and `Life`'s
        `composition` have to be in order to be compatible for terraformation.
- `lifespeed::Real = 0.2`: Distance `Life` can move in one timestep.
- `compositionmaxvalue::Int = 10`: Max possible value within `composition` vector.
- `compositionsize::Int = 10`: `length` of `composition` vector.
...

See also: [`galaxy_model_advanced`](@ref)
"""
function galaxy_model_basic(
    nplanets::Int;
    RNG::AbstractRNG = Random.default_rng(),
    extent::Tuple{<:Real,<:Real} = (1, 1), ## Size of space
    dt::Real = 1.0,
    interaction_radius::Union{Real,Nothing} = nothing, ## How close life and destination planet have to be to interact
    allowed_diff::Real = 3, ## How similar each element of life and destination planet have to be for terraformation
    lifespeed::Real = 0.2,
    compositionmaxvalue::Int = 10,
    compositionsize::Int = 10,
) ## Speed that life spreads

    isnothing(interaction_radius) && (interaction_radius = dt * lifespeed)

    galaxy_model_setup(
        :basic,
        @dict(
            nplanets,
            RNG,
            extent,
            dt,
            interaction_radius,
            allowed_diff,
            lifespeed,
            compositionmaxvalue,
            compositionsize
        )
    )

end

"""
    galaxy_model_advanced(; pos, vel, planetcompositions, <keyword arguments>)

Create an Agents.jl `ABM` to simulate life spreading throughout the galaxy. One of `pos`, 
`vel` or `planetcompositions` are required.

...
# Arguments
- `RNG::AbstractRNG = Random.default_rng()`: RNG object
- `extent::Tuple{<:Real,<:Real} = (1, 1)`: Bounds of the agent space
- `dt::Real = 1.0`: Model timestep
- `interaction_radius::Union{Real,Nothing} = nothing`: How close `Life` and destination
        `Planet` have to be to interact via `interacting_pairs`. Default is `dt*lifespeed`.
- `allowed_diff::Real = 3`: How similar each element of a `Planet`'s `composition` and 
        `Life`'s `composition` have to be in order to be compatible for terraformation.
- `lifespeed::Real = 0.2`: Distance `Life` can move in one timestep.
- `pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing`: `Planet` poss
- `vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing`: `Planet` vels
- `planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing`: `Planet` compositions
- `compositionmaxvalue::Int = 10`: Max possible value within `composition` vector.
- `compositionsize::Int = 10`: `length` of `composition` vector.
- `ool::Int = nothing`: ID of `Planet` to initialize `Life` on
...

See also: [`galaxy_model_basic`](@ref)
"""
function galaxy_model_advanced(;
    RNG::AbstractRNG = Random.default_rng(),
    extent::Tuple{<:Real,<:Real} = (1, 1),
    dt::Real = 1.0,
    interaction_radius::Union{Real,Nothing} = nothing,
    allowed_diff::Real = 3,
    lifespeed::Real = 0.2,
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing,
    compositionmaxvalue::Int = 10,
    compositionsize::Int = 10,
    ool::Int = nothing,
)

    isnothing(interaction_radius) && (interaction_radius = dt * lifespeed)

    galaxy_model_setup(
        :advanced,
        @dict(
            RNG,
            extent,
            dt,
            interaction_radius,
            allowed_diff,
            pos,
            vel,
            planetcompositions,
            compositionmaxvalue,
            compositionsize,
            lifespeed,
            ool
        )
    )

end

"""
Core function to set up the Planets (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function initialize_planets_unsafe(
    nplanets::Int,
    model::AgentBasedModel;
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing,
    compositionmaxvalue::Int = 10,
    compositionsize::Int = 10,
)

    ## Initialize arguments which are not provided 
    ## (flat random pos, no velocity, flat random compositions, 1 planet per system)
    isnothing(pos) && (
        pos = [
            (
                rand(RNG, Uniform(0, model.space.extend[1])),
                rand(RNG, Uniform(0, model.space.extend[2])),
            ) for _ = 1:nplanets
        ]
    )
    isnothing(vel) && (vel = [(0, 0) for _ = 1:nplanets])
    isnothing(planetcompositions) && (
        planetcompositions =
            [rand(RNG, 1:compositionmaxvalue, compositionsize) for _ = 1:nplanets]
    )

    for i = 1:nplanets
        pskwargs = Dict(
            :id => nextid(model),
            :pos => pos[i],
            :vel => vel[i],
            :composition => planetcompositions[i],
            :initialcomposition => planetcompositions[i],
            :alive => false,
            :claimed => false,
            :parentplanet => nothing,
            :parentlife => nothing,
            :parentcomposition => nothing,
            :ancestors => Vector{Planet}(undef, 0),
        )

        add_agent_pos!(Planet(; pskwargs...), model)

    end
    model

end

"""
Set up the Planets based only on `nplanets` (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function initialize_planets_basic!(
    nplanets::Int,
    model::AgentBasedModel;
    RNG::AbstractRNG = Random.default_rng(),
    compositionmaxvalue::Int = 10,
    compositionsize::Int = 10,
)

    nplanets < 1 && throw(ArgumentError("At least one planetary system required."))

    pos = nothing
    vel = nothing
    planetcompositions = nothing

    initialize_planets_unsafe(
        nplanets,
        model;
        @dict(RNG, pos, vel, planetcompositions, compositionmaxvalue, compositionsize)...,
    )

end

"""
    providedargs(args::Dict)

Return args, with pairs containing `nothing` values removed, as long as one pair has a 
non-`nothing` value. Used to check inputs of [`initialize_planets_advanced`](@ref)
(not user facing).
"""
function providedargs(args::Dict)

    providedargs = filter(x -> !isnothing(x.second), args)

    isempty(providedargs) ? throw(ArgumentError("one of $(keys(args)) must be provided")) :
    providedargs

end

"""
Return `true` if all args values have identical lengths. Used to check inputs of 
[`initialize_planets_advanced`](@ref) (not user facing).
"""
haveidenticallengths(args::Dict) =
    all(length(i.second) == length(args[collect(keys(args))[1]]) for i in args)

"""
Set up the Planets based only on `pos`, `vel`, or `planetcompositions` (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function initialize_planets_advanced!(
    model::AgentBasedModel;
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing,
    compositionmaxvalue::Int = 10,
    compositionsize::Int = 10,
)

    ## Validate user's args
    userargs = providedargs(@dict(pos, vel, planetcompositions))
    haveidenticallengths(userargs) ||
        throw(ArgumentError("provided arguments $(keys(userargs)) must all be same length"))

    ## Infered from userargs
    nplanets = length(userargs[collect(keys(userargs))[1]])

    initialize_planets_unsafe(
        nplanets,
        model;
        @dict(RNG, pos, vel, planetcompositions, compositionmaxvalue, compositionsize)...,
    )

end

"""
    compatibleplanets(planet, model)

Return `Vector{Planet}` where planet is compatible for terraformation (not user facing).
"""
function compatibleplanets(planet::Planet, model::ABM)

    candidateplanets = collect(values(filter(p -> isa(p.second, Planet), model.agents)))
    candidateplanets = collect(values(filter(
        p -> (p.alive == false) & (p.claimed == false) & (p.id != planet.id),
        candidateplanets,
    )))
    compositions = hcat([a.composition for a in candidateplanets]...)
    compositiondiffs = abs.(compositions .- planet.composition)
    compatibleindxs =
        findall(<=(model.allowed_diff), vec(maximum(compositiondiffs, dims = 1)))
    convert(Vector{Planet}, candidateplanets[compatibleindxs]) ## Returns Planets

end

"""
    nearestcompatibleplanet(planet, candidateplanets)

Return candidateplanet nearest to planet (not user facing).
"""
function nearestcompatibleplanet(planet::Planet, candidateplanets::Vector{Planet})

    length(candidateplanets) == 0 && throw(ArgumentError("candidateplanets is empty"))
    planetpositions = Array{Float64}(undef, 2, length(candidateplanets))
    for (i, a) in enumerate(candidateplanets)
        planetpositions[1, i] = a.pos[1]
        planetpositions[2, i] = a.pos[2]
    end
    idx, dist = nn(KDTree(planetpositions), collect(planet.pos))
    candidateplanets[idx] ## Returns Planet

end

"""
Core function to set up and spawn Life (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM;
    ancestors::Union{Nothing,Vector{Life}} = nothing,
)
    planet.alive = true
    planet.claimed = true ## This should already be true unless this is the first planet
    ## No ancestors, parentplanet, parentlife, parentcomposition
    candidateplanets = compatibleplanets(planet, model)
    if length(candidateplanets) == 0
        @warn "Life on Planet $(planet.id) has no compatible planets. It's the end of its line."
        destinationplanet = nothing
        vel = planet.pos .* 0.0
    else
        destinationplanet = nearestcompatibleplanet(planet, candidateplanets)
        vel = direction(planet, destinationplanet) .* model.lifespeed
    end

    args = Dict(
        :id => nextid(model),
        :pos => planet.pos,
        :vel => vel,
        :parentplanet => planet,
        :composition => planet.composition,
        :destination => destinationplanet,
        :ancestors => isnothing(ancestors) ? Planet[] : ancestors,
    ) ## Only "first" life won't have ancestors

    life = add_agent_pos!(Life(; args...), model)

    !isnothing(destinationplanet) && (destinationplanet.claimed = true) ## destination is only nothing if no compatible planets 
    ## NEED TO MAKE SURE THAT THE FIRST LIFE HAS PROPERTIES RECORDED ON THE FIRST PLANET

    model

end

"""
    mixcompositions(lifecomposition, planetcomposition)

Return rounded element-averaged composition (not user facing).
"""
function mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})
    ## Simple for now; Rounding goes to nearest even number
    convert(Vector{Int}, round.((lifecomposition .+ planetcomposition) ./ 2))
end

"""
    terraform!(life, planet, model)

Perform actions on `life` and `planet` associated with successful terraformation. Takes
existing `life` and terraforms an exsiting non-alive `planet`.
- Mix the composition of `planet` and `life`
- Update the `planet` to `alive=true`
- Update the `planet`'s ancestors, parentplanet, parentlife, and parentcomposition
- Call `spawnlife!` to send out life from `planet`.
"""
function terraform!(life::Life, planet::Planet, model::ABM)

    ## Modify destination planet properties
    planet.composition = mixcompositions(planet.composition, life.composition)
    planet.alive = true
    push!(planet.ancestors, life.parentplanet)
    planet.parentplanet = life.parentplanet
    planet.parentlife = life
    planet.parentcomposition = life.composition
    # planet.claimed = true ## Test to make sure this is already true beforehand

    spawnlife!(planet, model, ancestors = push!(life.ancestors, life)) ## This makes new life 

end

"""
    galaxy_model_step(model)

Custom `model_step` to be called by `Agents.step!`. Checks all `interacting_pairs`, and 
terraforms a `Planet` if a `Life` has reached its destination; then kills that `Life`.
"""
function galaxy_model_step!(model)
    ## I need to scale the interaction radius by dt and the velocity of life or else I can 
    ##   miss some interactions

    life_to_kill = Life[]
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, planet = typeof(a1) == Planet ? (a2, a1) : (a1, a2)
        if planet == life.destination
            terraform!(life, planet, model)
            push!(life_to_kill, life)
        end

    end

    for life in life_to_kill
        kill_agent!(life, model)
    end

end

## Fun with colors
# col_to_hex(col) = "#"*hex(col)
# hex_to_col(hex) = convert(RGB{Float64}, parse(Colorant, hex))
# mix_cols(c1, c2) = RGB{Float64}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)

end # module

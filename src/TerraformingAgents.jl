module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot
using Distributions: Uniform
using NearestNeighbors

export Planet, Life, galaxy_model_setup, galaxy_model_step!

"""
    random_agent([rng = Random.default_rng(),] A::Type, model)

Return random agent of type `A` (not user facing).
"""
function Agents.random_agent(rng::AbstractRNG, A::Type, model)
    agents = [k for (k, v) in model.agents if v isa A]
    if !isempty(agents)
        model[rand(rng, agents)]
    else
        error("model has no agents of type $A")
    end
end
Agents.random_agent(A::Type, model) = random_agent(Random.default_rng(), A, model)

magnitude(x) = sqrt(sum(x .^ 2))

"""
    direction(start, finish)

Return normalized direction from `start::AbstractAgent` to `finish::AbstractAgent` (not
user facing).
"""
direction(start::AbstractAgent, finish::AbstractAgent) = let δ = finish.pos .- start.pos
    δ ./ magnitude(δ)
end

Base.@kwdef mutable struct Planet <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}

    composition::Vector{Int} ## Represents the planet's genotype
    initialcomposition::Vector{Int} = composition ## Same as composition until it's terraformed

    alive::Bool = false
    ## True if any Life has this planet as its destination
    claimed::Bool = false

    # Properties of the process, but not the planet itself

    ## Planets that phylogenetically preceded this one
    ancestors::Vector{Planet} = Planet[]

    ## Planet that directly preceded this one
    parentplanet::Union{Planet, Nothing} = nothing
    parentlife::Union{<:AbstractAgent, Nothing} = nothing
    parentcomposition::Union{Vector{Int}, Nothing} = nothing
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    parentplanet::Planet
    composition::Vector{Int} ## Taken from parentplanet
    destination::Union{Planet, Nothing}
    ancestors::Vector{Life} ## Life agents that phylogenetically preceded this one
end

function random_positions(rng, (xmax, ymax), n)
    Ux = Uniform(0, xmax)
    Uy = Uniform(0, ymax)
    collect(zip(rand(rng, Ux, n), rand(rng, Uy, n))) :: Vector{NTuple{2, Float64}}
end

default_velocities(n) = fill((0.0, 0.0), n) :: Vector{NTuple{2, Float64}}

random_compositions(rng, maxcomp, compsize, n) = rand(rng, 1:maxcomp, compsize, n)

struct TerraformParameters
    extent::NTuple{2, Float64}
    dt::Float64
    lifespeed::Float64
    interaction_radius::Float64
    allowed_diff::Float64
    ool::Union{Int, Nothing}
    pos::Vector{NTuple{2, Float64}}
    vel::Vector{NTuple{2, Float64}}
    planetcompositions::Array{Int64, 2}

    function TerraformParameters(; extent::NTuple{2,<:Real}=(1.0, 1.0),
                                   dt::Real=10,
                                   lifespeed::Real=0.2,
                                   interaction_radius::Real=dt * lifespeed,
                                   allowed_diff::Real=2.0,
                                   ool::Union{Real, Nothing}=nothing,
                                   pos::Vector{<:NTuple{2, <:Real}},
                                   vel::Vector{<:NTuple{2, <:Real}},
                                   planetcompositions::Array{<:Integer, 2}
                                )

        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        new(extent, dt, lifespeed, interaction_radius, allowed_diff, ool, pos, vel,
            planetcompositions)
    end
end

function TerraformParameters(rng::AbstractRNG, nplanets::Int;
        extent=(1.0, 1.0), maxcomp=10, compsize=10,
        pos::Vector{<:NTuple{2, <:Real}} = random_positions(rng, extent, nplanets),
        vel::Vector{<:NTuple{2, <:Real}} = default_velocities(nplanets),
        planetcompositions::Array{<:Integer,2} = random_compositions(rng, maxcomp, compsize, nplanets),
        kwargs...
    )

    TerraformParameters(; extent, pos, vel, planetcompositions, kwargs...)
end

function TerraformParameters(rng::AbstractRNG;
        pos::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
        vel::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
        planetcompositions::Union{<:Array{<:Integer,2}, Nothing} = nothing,
        kwargs...
    )

    if !isnothing(pos)
        nplanets = length(pos)
    elseif !isnothing(vel)
        nplanets = length(vel)
    elseif !isnothing(planetcompositions)
        nplanets = size(planetcompositions, 2)
    else
        msg = "at least one of :pos, :vel or :planetcompositions must be provided as a kwarg"
        throw(ArgumentError(msg))
    end

    args = Dict{Symbol,Any}(kwargs)

    !isnothing(pos) && (args[:pos] = pos)
    !isnothing(vel) && (args[:vel] = vel)
    !isnothing(planetcompositions) && (args[:planetcompositions] = planetcompositions)

    TerraformParameters(rng, nplanets; args...)
end

nplanets(params::TerraformParameters) = length(params.pos)

"""
Set up the galaxy model (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function galaxy_model_setup(rng::AbstractRNG, params::TerraformParameters)
    space2d = ContinuousSpace(2; periodic = true, extend = params.extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life},
        space2d,
        properties = Dict(:dt => params.dt,
                          :interaction_radius => params.interaction_radius,
                          :allowed_diff => params.allowed_diff,
                          :lifespeed => params.lifespeed)
    )

    initialize_planets!(model, params)

    agent = isnothing(params.ool) ? random_agent(rng, Planet, model) : model.agents[params.ool]
    spawnlife!(agent, model)
    index!(model)
    model
end

galaxy_model_setup(params::TerraformParameters) = galaxy_model_setup(Random.default_rng(), params)

function galaxy_model_setup(rng::AbstractRNG, args...; kwargs...)
    galaxy_model_setup(rng, TerraformParameters(rng, args..., kwargs...))
end

"""
Core function to set up the Planets (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function initialize_planets!(model, params::TerraformParameters)
    for i = 1:nplanets(params)
        id = nextid(model)
        pos = params.pos[i]
        vel = params.vel[i]
        composition = params.planetcompositions[:, i]

        planet = Planet(; id, pos, vel, composition)

        add_agent_pos!(planet, model)
    end
    model
end

"""
    compatibleplanets(planet, model)

Return `Vector{Planet}` of `Planet`s compatible with `planet` for terraformation (not user
facing).
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

Return `Planet` within `candidateplanets` that is nearest to `planet `(not user facing).
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
Core function to set up and spawn `Life` (not user facing).

Called by [`galaxy_model_basic`](@ref) and [`galaxy_model_advanced`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM;
    ancestors::Vector{Life} = Life[],
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

    life = Life(;
        id = nextid(model),
        pos = planet.pos,
        vel = vel,
        parentplanet = planet,
        composition = planet.composition,
        destination = destinationplanet,
        ancestors
    ) ## Only "first" life won't have ancestors

    life = add_agent_pos!(life, model)

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
- Mix the `composition` of `planet` and `life`
- Update the `planet` to `alive=true`
- Update the `planet`'s `ancestors`, `parentplanet`, `parentlife`, and `parentcomposition`
- Call `spawnlife!` to send out `Life` from `planet`.
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

Custom `model_step` to be called by `Agents.step!`. Check all `interacting_pairs`, and
`terraform` a `Planet` if a `Life` has reached its destination; then kill that `Life`.
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

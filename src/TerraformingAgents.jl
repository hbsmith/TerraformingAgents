module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
# using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot
using Distributions: Uniform
using NearestNeighbors

export Planet, Life, galaxy_model_setup, galaxy_model_step!, GalaxyParameters

"""
    direction(start::AbstractAgent, finish::AbstractAgent)

Returns normalized direction from `start::AbstractAgent` to `finish::AbstractAgent` (not
user facing).
"""
direction(start::AbstractAgent, finish::AbstractAgent) = let δ = finish.pos .- start.pos
    δ ./ hypot(δ...)
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

"""
    All get passed to the ABM model as ABM model properties
"""
struct GalaxyProperties
    dt
    lifespeed
    interaction_radius
    allowed_diff
    ool
    pos
    vel
    planetcompositions

    function GalaxyProperties(
        dt::Real = 10,
        lifespeed::Real = 0.2,
        interaction_radius::Real = dt*lifespeed,
        allowed_diff::Real = 2.0,
        ool::Union{Vector{Int}, Int, Nothing} = nothing,
        pos::Vector{<:NTuple{2, <:Real}}
        vel::Vector{<:NTuple{2, <:Real}}
        planetcompositions::Array{<:Int, 2})

        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        new(dt, lifespeed, interaction_radius, allowed_diff, ool, pos, vel, planetcompositions)

    end
    
end

## Create with random pos, vel, planetcompositions
## Properties of randomized planetcompositions can be changed with new fields maxomp, compsize
function GalaxyProperties(nplanets::Int;
    maxcomp=10, 
    compsize=10,
    # pos=random_positions(rng, extent, nplanets),
    # vel=default_velocities(nplanets),
    # planetcompositions=random_compositions(rng, maxcomp, compsize, nplanets),
    kwargs...
)
#     ## Do I need the type definitions here if they're going to be invoked when GalaxyParameters is called below?
#     pos::Vector{<:NTuple{2, <:Real}} = random_positions(rng, extent, nplanets),
#     vel::Vector{<:NTuple{2, <:Real}} = default_velocities(nplanets),
#     planetcompositions::Array{<:Integer,2} = random_compositions(rng, maxcomp, compsize, nplanets),
#     kwargs...
# )
GalaxyProperties(; extent, pos, vel, planetcompositions, kwargs...)
GalaxyProperties(; extent, pos, vel, planetcompositions, kwargs...)
end

## Simply only require nplanets
function GalaxyProperties(nplanets::Int; kwargs...)
    ## Uses constructor above (with optional args of maxcomp, compsize) to randomize unprovided arguments
    println("nplanets::Int; kwargs...")
    GalaxyProperties(Random.default_rng(), nplanets; kwargs...) ## If it's ", kwargs..." instead of "; kwargs...", then I get an error from running something like this: TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0))
end

"""
    Other kwargs of ABM besides properties (which fall into GalaxyProperties above)
    Do I need this or can I wrap it into some other kwargs so that I don't have to copy and paste default kwargs?
"""
struct ABMkwargs
    scheduler
    rng
    warn

    function ABMkwargs(
        scheduler = Schedulers.fastest
        rng::AbstractRNG = Random.default_rng()
        warn = false)

        new(scheduler, rng, warn)

    end
end

"""
    All get passed into ContinuousSpace
"""
struct SpaceArgs
    extent
    spacing
    update_vel!
    periodic

    function SpaceArgs(
        extent::NTuple{2,<:Real}=(1.0, 1.0),
        spacing = min(extent...) / 10.0;
        update_vel! = defvel,
        periodic = true)

        new(extent, spacing, update_vel!, periodic)
    end

end

"""
    Do I need this? Idea would be to include only things which are needed during initializaiton, 
    e.g. initial_particles in this example: https://juliadynamics.github.io/Agents.jl/stable/examples/fractal_growth/
    But I think this could be handled by using the initial value of nplanets to create first batch of planets, 
    and if that number is later updated via slider
"""
struct GalaxyInits
end

struct GalaxyParameters
    extent::NTuple{2, Float64}
    dt::Float64
    lifespeed::Float64
    interaction_radius::Float64
    allowed_diff::Float64
    ool::Union{Int, Nothing}
    pos::Vector{NTuple{2, Float64}}
    vel::Vector{NTuple{2, Float64}}
    planetcompositions::Array{Int64, 2}

    ## Everything is optional here
    ## Only fields w/o defaults: pos, vel, planetcomposition
    function GalaxyParameters(; extent::NTuple{2,<:Real}=(1.0, 1.0),
                                   dt::Real=10,
                                   lifespeed::Real=0.2,
                                   interaction_radius::Real=dt * lifespeed,
                                   allowed_diff::Real=2.0,
                                   ool::Union{Real, Nothing}=nothing,
                                   pos::Vector{<:NTuple{2, <:Real}},
                                   vel::Vector{<:NTuple{2, <:Real}},
                                   planetcompositions::Array{<:Integer, 2}
                                )

        println("struct constructor")
        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        # @show stacktrace()
        # show(Base.stdout, MIME"text/plain"(), stacktrace())

        new(extent, dt, lifespeed, interaction_radius, allowed_diff, ool, pos, vel,
            planetcompositions)
    end
end

## Create with random pos, vel, planetcompositions
## Properties of randomized planetcompositions can be changed with new fields maxomp, compsize
function GalaxyParameters(rng::AbstractRNG, nplanets::Int;
        extent=(1.0, 1.0), 
        maxcomp=10, 
        compsize=10,
        pos=random_positions(rng, extent, nplanets),
        vel=default_velocities(nplanets),
        planetcompositions=random_compositions(rng, maxcomp, compsize, nplanets),
        kwargs...
    )
    #     ## Do I need the type definitions here if they're going to be invoked when GalaxyParameters is called below?
    #     pos::Vector{<:NTuple{2, <:Real}} = random_positions(rng, extent, nplanets),
    #     vel::Vector{<:NTuple{2, <:Real}} = default_velocities(nplanets),
    #     planetcompositions::Array{<:Integer,2} = random_compositions(rng, maxcomp, compsize, nplanets),
    #     kwargs...
    # )
    println("rng,nplanets")
    GalaxyParameters(; extent, pos, vel, planetcompositions, kwargs...)
end

## Simply only require nplanets
function GalaxyParameters(nplanets::Int; kwargs...)
    ## Uses constructor above (with optional args of maxcomp, compsize) to randomize unprovided arguments
    println("nplanets::Int; kwargs...")
    GalaxyParameters(Random.default_rng(), nplanets; kwargs...) ## If it's ", kwargs..." instead of "; kwargs...", then I get an error from running something like this: TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0))
end

# ## Simply only require nplanets
# function GalaxyParameters(;nplanets::Int, kwargs...)
#     ## Uses constructor above (with optional args of maxcomp, compsize) to randomize unprovided arguments
#     println(";nplanets::Int, kwargs...")
#     GalaxyParameters(Random.default_rng(), nplanets; kwargs...) ## If it's ", kwargs..." instead of "; kwargs...", then I get an error from running something like this: TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0))
# end

## Requires one of pos, vel, planetcompositions
## Would it be more clear to write this as 3 separate functions?
function GalaxyParameters(rng::AbstractRNG;
        pos::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
        vel::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
        planetcompositions::Union{<:Array{<:Integer,2}, Nothing} = nothing,
        kwargs...
    )

    println("rng;")

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

    ## Uses nplanets constructor for other artuments
    GalaxyParameters(rng, nplanets; args...)
end

nplanets(params::GalaxyParameters) = length(params.pos)

"""
    galaxy_model_setup(rng::AbstractRNG, params::GalaxyParameters)

Sets up the galaxy model.
"""
function galaxy_model_setup(rng::AbstractRNG, params::GalaxyParameters)
    space2d = ContinuousSpace(params.extent; periodic = true)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life},
        space2d,
        properties = Dict(:dt => params.dt,
                          :interaction_radius => params.interaction_radius,
                          :allowed_diff => params.allowed_diff,
                          :lifespeed => params.lifespeed)
    )

    initialize_planets!(model, params)

    agent = isnothing(params.ool) ? random_agent(model, x -> x isa Planet) : model.agents[params.ool]
    spawnlife!(agent, model)
    # index!(model)
    model
end

galaxy_model_setup(params::GalaxyParameters) = galaxy_model_setup(Random.default_rng(), params)

function galaxy_model_setup(rng::AbstractRNG, args...; kwargs...)
    galaxy_model_setup(rng, GalaxyParameters(rng, args..., kwargs...))
end

"""
    initialize_planets!(model, params::GalaxyParameters)

Adds Planets (not user facing).

Called by [`galaxy_model_setup`](@ref).
"""
function initialize_planets!(model, params::GalaxyParameters)
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
    compatibleplanets(planet::Planet, model::ABM)

Returns `Vector{Planet}` of `Planet`s compatible with `planet` for terraformation (not user
facing).
"""
function compatibleplanets(planet::Planet, model::ABM)
    function iscandidate((_, p))
        isa(p, Planet) && !p.alive && !p.claimed && p.id != planet.id
    end

    candidateplanets = collect(values(filter(iscandidate, model.agents)))
    compositions = hcat([a.composition for a in candidateplanets]...)
    compositiondiffs = abs.(compositions .- planet.composition)
    compatibleindxs =
        findall(<=(model.allowed_diff), vec(maximum(compositiondiffs, dims = 1)))

    ## Necessary in cased the result is empty
    convert(Vector{Planet}, candidateplanets[compatibleindxs]) ## Returns Planets

end

"""
    nearestcompatibleplanet(planet::Planet, candidateplanets::Vector{PLanet})

Returns `Planet` within `candidateplanets` that is nearest to `planet `(not user facing).
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
    spawnlife!(planet::Planet, model::ABM; ancestors::Vector{Life} = Life[])

Spawns `Life` (not user facing).

Called by [`galaxy_model_setup`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM;
    ancestors::Vector{Life} = Life[]
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
    mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})

Rounds element-averaged composition (not user facing).
"""
function mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})
    ## Simple for now; Rounding goes to nearest even number
    round.(Int, (lifecomposition .+ planetcomposition) ./ 2)
end

"""
    terraform!(life::Life, planet::Planet, model::ABM)

Performs actions on `life` and `planet` associated with successful terraformation. Takes
existing `life` and terraforms an exsiting non-alive `planet` (not user facing).
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

Custom `model_step` to be called by `Agents.step!`. Checks all `interacting_pairs`, and
`terraform`s a `Planet` if a `Life` has reached its destination; then kills that `Life`.
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

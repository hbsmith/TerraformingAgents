module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot
using NearestNeighbors

export 
Planet, 
Life,
galaxy_model_basic,
galaxy_model_advanced,
galaxy_model_step!

Agents.random_agent(model, A::Type{T}, RNG::AbstractRNG=Random.default_rng()) where {T<:AbstractAgent} = model[rand(RNG, [k for (k,v) in model.agents if v isa A])]
# Agents.random_agent(model, A::Type{T}) where {T<:AbstractAgent} = model[rand([k for (k,v) in model.agents if v isa A])]

magnitude(x::Tuple{<:Real,<:Real}) = sqrt(sum(x .^ 2))
direction(start::Agent, finish::Agent) = (finish.pos .- start.pos) ./ magnitude((finish.pos .- start.pos)) ## This is normalized

Base.@kwdef mutable struct Planet <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    
    # nplanets::Int # To simplify the initial logic, this will be limited to 1 
    composition::Vector{Int} ## I'll always make 10 planet compositions, but only use the first nplanets of them
    initialcomposition::Vector{Int} ## Same as composition until it's terraformed
    alive::Bool
    claimed::Bool
    
    ## Properties of the process, but not the planet itself
    ancestors::Vector{Int} #list of all planets that phylogenetically preceded this life
    parentplanet::Union{Int,Nothing} #id
    parentlife::Union{Int,Nothing} #id
    parentcomposition::Union{Vector{Int},Nothing} 
    
    ## Do I need the below? If planets disappear after certain length of time
    # age::Float64
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    parentplanet::Int #id; this is also the "type" of life
    composition::Vector{Int} # to simplify initial logic, this will be a single vector of length 10. Should this be the same as the planet composition?
    destination::Union{Nothing,Int} #id of destination planetarysystem
    ancestors::Vector{Int} #list of all life that phylogenetically preceded this life
    ## once life arrives at a new planet, life the agent just "dies"
end

function galaxy_model_setup(detail::Symbol, kwarg_dict::Dict)

    @unpack RNG, extent, dt, interaction_radius, allowed_diff, lifespeed = kwarg_dict

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            allowed_diff, 
            lifespeed))

    if detail == :basic 
        @unpack nplanets = kwarg_dict
        initialize_planetarysystems_basic!(model, nplanets; @dict(RNG)...)
    elseif detail == :advanced
        @unpack pos, vel, planetcompositions = kwarg_dict
        initialize_planetarysystems_advanced!(model; @dict(RNG, pos, vel, planetcompositions)...)
    else
        throw(ArgumentError("`detail` must be `:basic` or `:advanced`"))
    end
    
    # initialize_psneighbors!(model, ) # Add neighbor's within 
    # initialize_nearest_neighbor!(model) # Add nearest neighbor
    initialize_life!(random_agent(model, Planet, RNG), model)   
    index!(model)
    model

end

function galaxy_model_basic(
    nplanets::Int; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    allowed_diff::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    lifespeed::Real = 0.2) ## number of planets per star)

    galaxy_model_setup(:basic, @dict(nplanets, RNG, extent, dt, interaction_radius, allowed_diff, lifespeed))

end

function galaxy_model_advanced(; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    allowed_diff::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    lifespeed::Real = 0.2,
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    galaxy_model_setup(:advanced, @dict(RNG, extent, dt, interaction_radius, allowed_diff, pos, vel, planetcompositions, lifespeed))

end

function providedargs(args::Dict) 
    
    providedargs = filter(x -> !isnothing(x.second), args)

    isempty(providedargs) ? throw(ArgumentError("one of $(keys(args)) must be provided")) : providedargs

end 

haveidenticallengths(args::Dict) = all(length(i.second) == length(args[collect(keys(args))[1]]) for i in args)

function initialize_planetarysystems_unsafe!(
    model::AgentBasedModel,
    nplanets::Int; 
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    ## Initialize arguments which are not provided 
    ## (flat random pos, no velocity, flat random compositions, 1 planet per system)
    isnothing(pos) && (pos = [Tuple(rand(RNG,2)) for _ in 1:nplanets])
    isnothing(vel) && (vel = [(0,0) for _ in 1:nplanets])
    isnothing(planetcompositions) && (planetcompositions = [rand(RNG,1:10,10) for _ in 1:nplanets])

    # Add PlanetarySystem agents
    for i in 1:nplanets
        
        pskwargs = Dict(:id => nextid(model),
                    :pos => pos[i],
                    :vel => vel[i],
                    :composition => planetcompositions[i],
                    :initialcomposition => planetcompositions[i],
                    :alive => false,
                    :claimed => false,
                    :parentplanet => nothing,
                    :parentlife => nothing,
                    :parentcomposition => nothing,
                    :ancestors => Vector{Int}(undef,0))
        
        add_agent_pos!(Planet(;pskwargs...), model)
    
    end
    model

end

function initialize_planetarysystems_basic!(
    model::AgentBasedModel,
    nplanets::Int; 
    RNG::AbstractRNG = Random.default_rng())

    nplanets < 1 && throw(ArgumentError("At least one planetary system required."))

    pos=nothing
    vel=nothing 
    planetcompositions=nothing

    initialize_planetarysystems_unsafe!(model, nplanets; @dict(RNG, pos, vel, planetcompositions)...)    

end

function initialize_planetarysystems_advanced!(
    model::AgentBasedModel; 
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    ## Validate user's args
    userargs = providedargs(@dict(pos, vel, planetcompositions))
    haveidenticallengths(userargs) || throw(ArgumentError("provided arguments $(keys(userargs)) must all be same length"))
    
    ## Infered from userargs
    nplanets = length(userargs[collect(keys(userargs))[1]])

    initialize_planetarysystems_unsafe!(model, nplanets; @dict(RNG, pos, vel, planetcompositions)...)    

end

function compatibleplanetids(planet::Planet, model::ABM)

    candidateplanets = filter(p->p.second.alive==false & isa(p.second, Planet) & p.second.claimed==false, model.agents) ## parentplanet won't be here because it's already claimed
    ncandidateplanets = length(candidateplanets)

    planetids = Vector{Int}(undef,ncandidateplanets)
    _planetvect = Vector{Int}(undef,ncandidateplanets)
    for (i,a) in enumerate(values(candidateplanets))
        _planetvect[i] = a.planetcompositions
        planetids[i] = a.id
    end    
    allplanetcompositions = hcat(_planetvect...)
    compositiondiffs = abs.(allplanetcompositions .- planet.composition)
    compatibleindxs = findall(<=(model.llowed_diff),maximum(compositiondiffs, dims=1)) ## No element can differ by more than threshold
    planetids[compatibleindxs]

end

function nearestcompatibleplanet(planet::Planet, compatibleplanetids::Vector{Int}, model::ABM)

    planetpositions = Array{Real}(undef,2,length(compatibleplanetids))
    for (i,id) in enumerate(compatibleplanetids)
        planetpositions[1,i] = model.agent[id].pos[1]
        planetpositions[2,i] = model.agent[id].pos[2]
    end
    idx, dist = nn(KDTree(planetpositions),collect(planet.pos)) ## I need to make sure the life is initialized first with position
    compatibleplanetids[idx]

end

function spawnlife!(planet::Planet, model::ABM; ancestors::Union{Nothing,Vector{Int}}=nothing) ## First life is weird because it inherits from planet without changing planet and has no ancestors
## Design choice is to modify planet and life together since the life is just a reflection of the planet anyways
    planet.alive = true
    planet.claimed = true ## This should already be true unless this is the first planet
    ## No ancestors, parentplanet, parentlife, parentcomposition
    destinationplanet =  nearestcompatibleplanet(planet, compatibleplanetids(planet,model), model)

    args = Dict(:id => nextid(model),
                :pos => planet.pos,
                :vel => direction(planet,destinationplanet) .* model.lifespeed,
                :parentplanet => planet.id,
                :parentcomposition => planet.composition,
                :destination => destinationplanet,
                :ancestors => isnothing(ancestor) ? Int[] : ancestors) ## Only "first" life won't have ancestors

    life = add_agent_pos!(Life(;args...), model)
    
    destinationplanet.claimed = true
    model

end

## Life which has spawned elsewhere merging with an uninhabited (ie dead) planet
function terraform!(life::Life, planet::Planet)

    ## Modify destination planet properties
    planet.composition = mixcompositions(planet.compositon,life.composition)
    planet.alive = true
    push!(planet.ancestors, life.parentplanet)
    planet.parentplanet = life.parentplanet
    planet.parentlife = life.id 
    planet.parentcomposition = life.composition
    # planet.claimed = true ## Test to make sure this is already true beforehand

    spawnlife!(planet, model, ancestors = push!(life.ancestors, life.id)) ## This makes new life 
    println("terraformed $(planet.id) from $(life.id)")
    kill_agent!(life, model)

end

function approaching_planet(life::Life, planet::Planet)

    lifesRelativePos = life.pos .- planet.pos
    lifesRelativeVel = life.vel

    dot(lifesRelativePos,lifesRelativeVel) >= 0 ? false : true
end

function galaxy_model_step!(model)
    ## Interaction radius has to account for the velocity of life and the size of dt to ensure interaction
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, planet = typeof(a1) == Planet ? (a2, a1) : (a1, a2)
        life.parentplanet == planet.id && return ## don't accidentally interact with the parent planet
        approaching_planet(life, planet) && is_compatible(life, planet, model.allowed_diff) ? terraform!(life, planet) : return
    end
end

## TO DO 10/2/2020
## - MAKE CONSISTENT HOW I KEEP TRACK OF ancestors
## - MAKE CONSISTENT PASSING INSTANCES VS IDS, KEEPING IN MIND INSTANCES WILL DIE AND I WANT THEM TO BE GARBAGE COLLECTED
## - UPDATE TESTS

#= I should probably have a function that just scales my planet compositions 
accross the color spectrum instead of baking colors in as the compositions
themselves
=#

col_to_hex(col) = "#"*hex(col)
hex_to_col(hex) = convert(RGB{Float64}, parse(Colorant, hex))
mix_cols(c1, c2) = RGB{Float64}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)

# function model_step!(model)
#     for (a1, a2) in interacting_pairs(model, 0.2, :types)
        
#         if typeof(a1) == PlanetarySystem
#             a1.nearestlife = a2.id
        
#         elseif typeof(a1) == Life
#             a2.nearestlife = a1.id
        
#         end
        
#         # println(a2)
# #         elastic_collision!(a1, a2, :mass)
#     end
# end

# function sir_agent_step!(agent, model)
#     move_agent!(agent, model, model.dt)
#     update!(agent) # store information in life agent of it terraforming?
#     recover_or_die!(agent, model)
# end

## COMMENTING OUT EVEYRTHING BELOW FOR TESTING

# agent_step!(agent, model) = move_agent!(agent, model, model.dt)

# modelparams = Dict(:RNG => MersenneTwister(1236),
#                    : => .45,
#                    :dt => 0.1)

# model = galaxy_model(;modelparams...)

# model_colors(a) = typeof(a) == PlanetarySystem ? "#2b2b33" : "#338c54"

# e = model.space.extend
# anim = @animate for i in 1:2:100
#     p1 = plotabm(
#         model,
#         as = 5,
#         ac = model_colors,
#         showaxis = false,
#         grid = false,
#         xlims = (0, e[1]),
#         ylims = (0, e[2]),
#     )

#     title!(p1, "step $(i)")
#     step!(model, agent_step!, galaxy_model_step!, 2)
# end


# animation_path = "../output/animation/"
# if !ispath(animation_path)
#     mkpath(animation_path)
# end

# gif(anim, joinpath(animation_path,"terraform_test_death1.gif"), fps = 25)

end # module

#### Do all the calculations for nearest neighbors at the begining if the planetary systems don't move_agent
# - Otherwise, do them at each step if they do move
# - don't go back to planet you came from or planet that already has your life on it
# - fix velocity so that you go every direction at same speed and doesn't depend on how far away your target is. 

#### Could introduce time lag between terraforming and spreading, but I'll wait on this
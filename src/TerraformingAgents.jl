module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot

export 
PlanetarySystem, 
Life,
galaxy_model_basic,
galaxy_model_advanced,
galaxy_model_step!

Agents.random_agent(model, A::Type{T}, RNG::AbstractRNG=Random.default_rng()) where {T<:AbstractAgent} = model[rand(RNG, [k for (k,v) in model.agents if v isa A])]
# Agents.random_agent(model, A::Type{T}) where {T<:AbstractAgent} = model[rand([k for (k,v) in model.agents if v isa A])]

magnitude(x::Tuple{<:Real,<:Real}) = sqrt(sum(x .^ 2))

Base.@kwdef mutable struct PlanetarySystem <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    
    nplanets::Int # To simplify the initial logic, this will be limited to 1 
    planetcompositions::Vector{Vector{Int}} ## I'll always make 10 planet compositions, but only use the first nplanets of them
    alive::Bool
    
    ## Properties of the process, but not the planet itself
    parentplanet::Union{Int,Nothing} #id
    parentlife::Union{Int,Nothing} #id
    parentcomposition::Union{Vector{Int},Nothing} 
    nearestps::Union{Int,Nothing}
    neighbors::Union{Vector{Int}}
    
    ## Do I need the below?
    # age::Float64
    # originalcomposition::Vector{Vector{Int64}}
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    parentplanet::Int #id; this is also the "type" of life
    parentcomposition::Vector{Int} # to simplify initial logic, this will be a single vector of length 10
    destination::Int #id of destination planetarysystem
    ## once life arrives at a new planet, life the agent just "dies"
end

function galaxy_model_setup(detail::Symbol, kwarg_dict::Dict)

    @unpack RNG, extent, dt, interaction_radius, similarity_threshold, psneighbor_radius, lifespeed = kwarg_dict

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{PlanetarySystem,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            similarity_threshold, 
            psneighbor_radius))

    if detail == :basic 
        @unpack nplanetarysystems, nplanetspersystem = kwarg_dict
        initialize_planetarysystems_basic!(model, nplanetarysystems; @dict(RNG, nplanetspersystem)...)
    elseif detail == :advanced
        @unpack pos, vel, planetcompositions = kwarg_dict
        initialize_planetarysystems_advanced!(model; @dict(RNG, pos, vel, planetcompositions)...)
    else
        throw(ArgumentError("`detail` must be `:basic` or `:advanced`"))
    end
    
    initialize_psneighbors!(model, psneighbor_radius) # Add neighbor's within psneighbor_radius
    initialize_nearest_neighbor!(model) # Add nearest neighbor
    initialize_life!(random_agent(model, PlanetarySystem, RNG), model, lifespeed)   
    index!(model)
    
    return model

end

function galaxy_model_basic(
    nplanetarysystems::Int; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    psneighbor_radius::Real = 0.2, ## distance threshold used to decide where to send life from parent planet
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    similarity_threshold::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    nplanetspersystem::Int = 1,
    lifespeed::Real = 0.2) ## number of planets per star)

    galaxy_model_setup(:basic, @dict(nplanetarysystems, RNG, extent, dt, psneighbor_radius, interaction_radius, similarity_threshold, nplanetspersystem, lifespeed))

end

function galaxy_model_advanced(; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    psneighbor_radius::Real = 0.2, ## distance threshold used to decide where to send life from parent planet
    interaction_radius::Real = 1e-4, ## how close life and destination planet have to be to interact
    similarity_threshold::Real = 0.5, ## how similar life and destination planet have to be for terraformation
    lifespeed::Real = 0.2,
    pos::Union{Nothing,AbstractArray{Tuple{<:Real,<:Real}}} = nothing,
    vel::AbstractArray{Tuple{<:Real,<:Real}} = nothing,
    planetcompositions::Vector{Vector{Vector{Int}}} = nothing)

    galaxy_model_setup(:advanced, @dict(RNG, extent, dt, psneighbor_radius, interaction_radius, similarity_threshold, pos, vel, planetcompositions, lifespeed))

end

function providedargs(args::Dict) 
    
    providedargs = filter(x -> !isnothing(x.second), args)

    isempty(providedargs) ? throw(ArgumentError("one of $(keys(args)) must be provided")) : providedargs

end 

haveidenticallengths(args::Dict) = all(length(i.second) == length(args[collect(keys(args))[1]]) for i in args)

function initialize_planetarysystems_unsafe!(
    model::AgentBasedModel,
    nplanetarysystems::Int; 
    RNG::AbstractRNG = Random.default_rng(),
    nplanetspersystem::Int = 1,  ## Not used if planetcompositions provided
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Vector{Int}}}} = nothing)

    ## Initialize arguments which are not provided 
    ## (flat random pos, no velocity, flat random compositions, 1 planet per system)
    isnothing(pos) && (pos = [Tuple(rand(RNG,2)) for _ in 1:nplanetarysystems])
    isnothing(vel) && (vel = [(0,0) for _ in 1:nplanetarysystems])
    isnothing(planetcompositions) && (planetcompositions = [[rand(RNG,1:10,nplanetspersystem)] for _ in 1:nplanetarysystems])

    # Add PlanetarySystem agents
    for i in 1:nplanetarysystems
        
        pskwargs = Dict(:id => nextid(model),
                    :pos => pos[i],
                    :vel => vel[i],
                    :nplanets => length(planetcompositions[i]),
                    :planetcompositions => planetcompositions[i],
                    :alive => false,
                    :parentplanet => nothing,
                    :parentlife => nothing,
                    :parentcomposition => nothing,
                    :nearestps => nothing,
                    :neighbors => Vector{Int}(undef,0))
        
        add_agent_pos!(PlanetarySystem(;pskwargs...), model)
    
    end

end

function initialize_planetarysystems_basic!(
    model::AgentBasedModel,
    nplanetarysystems::Int; 
    RNG::AbstractRNG = Random.default_rng(),
    nplanetspersystem::Int = 1)

    nplanetarysystems < 1 && throw(ArgumentError("At least one planetary system required."))

    pos=nothing
    vel=nothing 
    planetcompositions=nothing

    initialize_planetarysystems_unsafe!(model, nplanetarysystems; @dict(RNG, nplanetspersystem, pos, vel, planetcompositions)...)    

end

function initialize_planetarysystems_advanced!(
    model::AgentBasedModel; 
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Vector{Int}}}} = nothing)

    ## Validate user's args
    userargs = providedargs(@dict(pos, vel, planetcompositions))
    haveidenticallengths(userargs) || throw(ArgumentError("provided arguments $(keys(userargs)) must all be same length"))
    
    ## Infered from userargs
    nplanetarysystems = length(userargs[collect(keys(userargs))[1]])

    initialize_planetarysystems_unsafe!(model, nplanetarysystems; @dict(RNG, pos, vel, planetcompositions)...)    

end

function initialize_psneighbors!(model::AgentBasedModel, radius::Float64)
    for (a1, a2) in interacting_pairs(model, radius, :all)
        push!(a1.neighbors, a2.id)
        push!(a2.neighbors, a1.id)
        # println("psneighbors: ",a1,a2)
    end
end

function initialize_nearest_neighbor!(model::AgentBasedModel) #, extent::NTuple{2,Union{Float64,Int}})
    for agent in values(model.agents)
        agent.nearestps = nearest_neighbor(agent, model, magnitude(model.space.extend)).id
        # println("nearest neighbor: ",agent.id, agent.nearestps)
    end
end

function initialize_life!(parentps::PlanetarySystem, model::AgentBasedModel, speed::Real)

    pos = parentps.pos
    
    if length(parentps.neighbors)>0
        for neighborid in parentps.neighbors

            # println(parentps.neighbors)
            direction = (model.agents[neighborid].pos .- pos)
            direction_normed = direction ./ magnitude(direction)

            largs = Dict(:id => nextid(model),
                        :pos => pos,
                        :vel => direction_normed .* speed)
            
            lkwargs = Dict(:parentplanet => parentps.id,
                        :parentcomposition => parentps.planetcompositions[1],
                        :destination => neighborid)
            
            add_agent_pos!(Life(;merge(largs,lkwargs)...), model)
        end
    else ## Go to nearest star, not neighbor stars

        direction = (model.agents[parentps.nearestps].pos .- pos)
        direction_normed = direction ./ magnitude(direction)

        largs = Dict(:id => nextid(model),
                    :pos => pos,
                    :vel => direction_normed .* speed)
            
        lkwargs = Dict(:parentplanet => parentps.id,
                    :parentcomposition => parentps.planetcompositions[1],
                    :destination => parentps.nearestps)
        
        add_agent_pos!(Life(;merge(largs,lkwargs)...), model)

    end


end

function is_compatible(life::Life, ps::PlanetarySystem, threshold::Float64)

    ## Placeholder
    return true

end

function terraform!(life::Life, ps::PlanetarySystem)
    ps.alive = true
    ps.parentplanet = life.parentplanet
    ps.parentcomposition = life.parentcomposition
    println("terraformed $(ps.id) from $(life.id)")
    ## Change ps composition based on life parent composition and current planet composition
    # ps.planetcompositions[1] = mix_compositions(life, ps)

end

function approaching_planet(life::Life, ps::PlanetarySystem)

    lifesRelativePos = life.pos .- ps.pos
    lifesRelativeVel = life.vel

    dot(lifesRelativePos,lifesRelativeVel) >= 0 ? false : true
end

function galaxy_model_step!(model)
    ## Interaction radius has to account for the velocity of life and the size of dt to ensure interaction
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, ps = typeof(a1) == PlanetarySystem ? (a2, a1) : (a1, a2)
        life.parentplanet == ps.id && return ## don't accidentally interact with the parent planet
        approaching_planet(life, ps) && is_compatible(life, ps, model.similarity_threshold) ? terraform!(life, ps) : return
        kill_agent!(life, model)
    end
end


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
#                    :psneighbor_radius => .45,
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

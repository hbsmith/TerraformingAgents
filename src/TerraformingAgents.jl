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
    
    
    ## Do I need the below?
    # age::Float64
    # originalcomposition::Vector{Vector{Int64}}
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
    initialize_nearest_neighbor!(model) # Add nearest neighbor
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

# function initialize_psneighbors!(model::AgentBasedModel, radius::Float64)
#     for (a1, a2) in interacting_pairs(model, radius, :all)
#         push!(a1.neighbors, a2.id)
#         push!(a2.neighbors, a1.id)
#         # println("psneighbors: ",a1,a2)
#     end
# end

# function initialize_nearest_neighbor!(model::AgentBasedModel) #, extent::NTuple{2,Union{Float64,Int}})
#     for agent in values(model.agents)
#         agent.nearestps = nearest_neighbor(agent, model, magnitude(model.space.extend)).id
#         # println("nearest neighbor: ",agent.id, agent.nearestps)
#     end
# end

# function findNearestDeadNeighbor(ps::PlanetarySystem)

#     return true

# end

# function neighboringTerraformCandidates(ps::PlanetarySystem, model::ABM, threshold::Real)

#     ## This can return >1 candidate
#     ## Two ways to do this 
#     ## 1. Use pre-calculatd ps.neighbors ids
#     # candidates = filter(n -> (model.agents[n].alive==false) & (ps.claimed==false), ps.neighbors) ## and ps.claimed == false


#     ## 2. use model.neighbor_radius and calcuate neighbors


#     ## Don't use neighbors, look at all other planets existing
#     candidatePlanets = filter(p->p.second.alive==true & isa(p.second, PlanetarySystem) & ps.claimed==false, model.agents)
#     candidatePlanetIds = 

#     diff = compositions.-testpscomp
#     maxdiffs = maximum(abs.(diff),dims=1)
#     findall(<(3),maxdiffs)

#     nn(KDTree(poss),poss[:,1])


# end

function compatibleplanetids(life::Life, model::ABM)

    candidateplanets = filter(p->p.second.alive==false & isa(p.second, Planet) & p.second.claimed==false, model.agents) ## parentplanet won't be here because it's already claimed
    ncandidateplanets = length(candidateplanets)

    planetids = Vector{Int}(undef,ncandidateplanets)
    _planetvect = Vector{Int}(undef,ncandidateplanets)
    for (i,a) in enumerate(values(candidateplanets))
        _planetvect[i] = a.planetcompositions
        planetids[i] = a.id
    end    
    allplanetcompositions = hcat(_planetvect...)
    compositiondiffs = abs.(allplanetcompositions .- life.composition)
    compatibleindxs = findall(<=(model.llowed_diff),maximum(compositiondiffs, dims=1)) ## No element can differ by more than threshold
    planetids[compatibleindxs]

end

function nearestcompatibleplanet(life::Life, compatibleplanetids::Vector{Int}, model::ABM)

    planetpositions = Array{Real}(undef,2,length(compatibleplanetids))
    for (i,id) in enumerate(compatibleplanetids)
        planetpositions[1,i] = model.agent[id].pos[1]
        planetpositions[2,i] = model.agent[id].pos[2]
    end
    idx, dist = nn(KDTree(planetpositions),collect(life.pos)) ## I need to make sure the life is initialized first with position
    compatibleplanetids[idx]

end


# function findNearestTerraformingCandidates(ps::PlanetarySystem, life::Life, model::AgentBasedModel)

#     viableDestinations = Int64[]
#     dibsedPlanets = Int64[]
#     # openPlanets = setdiff(ps.neighbors)
#     ## Need 3 filters
#     ## 1. planets within neighbor radius 
#     ## 2. compatible planets 
#     ## 3. unclaimed planets 

#     ## If at any point during 1->2->3 are no planets, find nearest planet which is compatible and unclaimed


#     if deadNeighbors(ps)
#         for newps in deadNeighbors(ps)
#             areCompatabile(ps,newps) && noDibs() ## Planets must be compatable and not destinations by current life
#         end
#     else
#         findNearestDeadNeighbors()
#         findNearestCompatibleNeighbors()
    

#     candidate_neighbors = filter(n->model.agents[n].alive==false, ps.neighbors)
#     length(candidate_neighbors) > 0 && return candidate_neighbors
#     ps.nearestps.alive || return Int64[ps.nearestps]
#     return Int64[findNearestDeadNeighbor(ps)]

#     ## NEED TO CHECK if any planets are current destinations

#     #= Should I have life avoid going to planets that are closer to other planets with life? 
#         Or will this natrually be taken care of because those would be destinations if that
#         was the case?
    
#     =#

#     #= pseduocode
#     initial_candidates = filter(dead, neighbors_within_neighborradius) OR filter(dead, nearest_neighbor)

#     ## Decide whether or not to go towards a planet beforehand, based on compatability? Or wait till
#         you head towards the planet and interact to decide whether or not your compatible? Logically
#         you should figure out beforehand if you're compatible, but maybe there's a limit on how 
#         confident you are in your assessment based on how far away you are and how different the
#         candidate planet is. This also seems too complicated though. So maybe for now you have
#         perfect knowledge of candidate planets before you leave, as long as they're within a certain
#         distance. And you only go towards compatible planets

#     ## Filter by dead candidates or unvisited candidates? All dead will be unvisited,
#         but unvisited may include alive planets settled by other related life. Maybe decided
#         based on how related the life is whether to try and settle or not? So if it's your immeadiate
#         family or first cousin don't settle for example. But this gets too complicated so let's stick
#         to settling dead planets only.

#     ## visited_planets == [ps.id for ps in filter(x->x.alive==true, filter(p->isa(p, PlanetarySystem), model.agents))

#     for ps in filter(dead, initial_candidates)
#         direction = (model.agents[ps.id].pos .- pos)
#         direction_normed = direction ./ magnitude(direction)
#         :vel => direction_normed .* model.lifespeed
#     =#

# end


function initialize_life!(planet::Planet, model::AgentBasedModel; ancestors::Union{Nothing,Vector{Int}}=nothing) ## Ancestor can't be the agent itself because it dies

    ## Update planet properties
    planet.claimed = true
    planet.alive = true

    ## Create Life without destination/velocity
    args = Dict(:id => nextid(model),
                :pos => planet.pos,
                :vel => (0.0,0.0),
                :parentplanet => planet.id,
                :parentcomposition => planet.composition,
                :destination => nothing,
                :ancestors => isnothing(ancestor) ? Int[] : ancestors) ## Only "first" life won't have ancestors

    life = add_agent_pos!(Life(;args...), model)
    life.destination = nearestcompatibleplanet(life, compatibleplanetids(life,model), model) ## return planet itself
    direction = (life.destination.pos .- life.pos)
    direction_normed = direction ./ magnitude(direction)
    life.vel = direction_normed .* model.lifespeed
    model

    # for psid in neighboringTerraformCandidates()

    #     direction = (model.agents[psid].pos .- pos)
    #     direction_normed = direction ./ magnitude(direction)

    #     largs = Dict(:id => nextid(model),
    #                 :pos => pos,
    #                 :vel => direction_normed .* model.lifespeed)
        
    #     lkwargs = Dict(:parentplanet => planet.id,
    #                 :parentcomposition => planet.composition[1],
    #                 :destination => psid)
       
    #     model.agents[psid].claimed = true
    #     add_agent_pos!(Life(;merge(largs,lkwargs)...), model)

    # end

    # ########
    
    # if length(planet.neighbors)>0
    #     for neighborid in planet.neighbors

    #         # println(parentps.neighbors)
    #         direction = (model.agents[neighborid].pos .- pos)
    #         direction_normed = direction ./ magnitude(direction)

    #         largs = Dict(:id => nextid(model),
    #                     :pos => pos,
    #                     :vel => direction_normed .* model.lifespeed)
            
    #         lkwargs = Dict(:parentplanet => planet.id,
    #                     :parentcomposition => planet.composition[1],
    #                     :destination => neighborid)
            
    #         add_agent_pos!(Life(;merge(largs,lkwargs)...), model)
    #     end
    # else ## Go to nearest star, not neighbor stars

    #     direction = (model.agents[planet.nearestps].pos .- pos) ## This won't work if nearest is a star you've been to
    #     direction_normed = direction ./ magnitude(direction)

    #     largs = Dict(:id => nextid(model),
    #                 :pos => pos,
    #                 :vel => direction_normed .* model.lifespeed)
            
    #     lkwargs = Dict(:parentplanet => planet.id,
    #                 :parentcomposition => planet.composition[1],
    #                 :destination => planet.nearestps)
        
    #     add_agent_pos!(Life(;merge(largs,lkwargs)...), model)

    # end


end

function terraform!(life::Life, planet::Planet)
    # planet.alive = true
    # planet.parentplanet = life.parentplanet
    # planet.parentcomposition = life.composition
    # planet.parentlife = life.id
    initialize_life!(planet, model, ancestors = push!(life.ancestors, ife.id))
    push!(planet.ancestors, life.parentplanet) ## Need to reevaluate how i think about ancestors and using life vs planet for that--go back to my original hypothesis points
    println("terraformed $(planet.id) from $(life.id)")
    ## Change ps composition based on life parent composition and current planet composition
    # ps.planetcompositions[1] = mix_compositions(life, ps)

end

function approaching_planet(life::Life, ps::Planet)

    lifesRelativePos = life.pos .- ps.pos
    lifesRelativeVel = life.vel

    dot(lifesRelativePos,lifesRelativeVel) >= 0 ? false : true
end

function galaxy_model_step!(model)
    ## Interaction radius has to account for the velocity of life and the size of dt to ensure interaction
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, ps = typeof(a1) == Planet ? (a2, a1) : (a1, a2)
        life.parentplanet == ps.id && return ## don't accidentally interact with the parent planet
        approaching_planet(life, ps) && is_compatible(life, ps, model.allowed_diff) ? terraform!(life, ps) : return
        kill_agent!(life, model)
        initialize_life!(ps, model)
    end
end


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
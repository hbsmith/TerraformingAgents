module TerraformingAgents;

# include("utilities.jl")

using Agents, Random, Printf
using Statistics: cor
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot, diag, issymmetric, tril!
using Distributions: Uniform
using NearestNeighbors
using Distances
using DataFrames
using StatsBase
using StaticArrays

export Planet, 
       Life, 
       galaxy_model_setup, 
       galaxy_agent_step_spawn_on_terraform!, 
       galaxy_agent_step_spawn_at_rate!, 
       galaxy_agent_direct_step!, 
       galaxy_model_step!, 
       GalaxyParameters, 
       compositionally_similar_planets,
       nearest_k_planets,
       planets_in_range,
       nearest_planet,
       most_similar_planet,
       filter_agents, 
       crossover_one_point, 
       horizontal_gene_transfer, 
       split_df_agent, 
       clean_df

"""
    direction(start::AbstractAgent, finish::AbstractAgent)

Return normalized direction from `start::AbstractAgent` to `finish::AbstractAgent`.
"""
direction(start::AbstractAgent, finish::AbstractAgent) = let δ = finish.pos .- start.pos
    δ ./ hypot(δ...)
end

"""
    distance(p1, p2)

Return euclidean distance between two points.
"""
distance(p1,p2) = hypot((p1 .- p2)...)

"""
    Planet{D} <: AbstractAgent

One of the two Agent types. Can be terraformed by `Life`. Exists in space of dimension `D`.

See also [`Life`](@ref)
"""
## Ideally all arguments inside of Planet and Life would be Float64, or Vector{Float64}, but 
##  apparently when using Parametric functions you can't coerce Ints into Floats for example.
## So that's super annoying. See: https://github.com/JuliaLang/julia/issues/35053
Base.@kwdef mutable struct Planet{D} <: AbstractAgent
    id::Int
    pos::SVector{D, Float64}
    vel::SVector{D, Float64}

    composition::Vector{<:Real} ## Represents the planet's genotype
    initialcomposition::Vector{<:Real} = copy(composition) ## Same as composition until it's terraformed

    alive::Bool = false
    claimed::Bool = false ## True if any Life has this planet as its destination
    spawn_threshold = 0
    candidate_planets::Vector{Planet} = Planet[]

    parentplanets::Vector{Planet} = Planet[] ## List of Planet objects that are this planet's direct parent
    parentlifes::Vector{<:AbstractAgent} = AbstractAgent[] ## List of Life objects that are this planet's direct parent
    parentcompositions::Vector{<:Vector{<:Real}} = Vector{Float64}[] ## List of compositions of the direct life parent compositions at time of terraformation
end
function Base.show(io::IO, planet::Planet{D}) where {D}
    s = "Planet 🪐 in $(D)D space with properties:."
    s *= "\n id: $(planet.id)"
    s *= "\n pos: $(planet.pos)"
    s *= "\n vel: $(planet.vel)"
    s *= "\n composition: $(planet.composition)"
    s *= "\n initialcomposition: $(planet.initialcomposition)"
    s *= "\n alive: $(planet.alive)"
    s *= "\n claimed: $(planet.claimed)"
    s *= "\n parentplanets (†‡): $(length(planet.parentplanets) == 0 ? "No parentplanet" : planet.parentplanets[end].id)"
    s *= "\n parentlifes (†‡): $(length(planet.parentlifes) == 0 ? "No parentlife" : planet.parentlifes[end].id)"
    s *= "\n parentcompositions (‡): $(length(planet.parentcompositions) == 0 ? "No parentcomposition" : planet.parentcompositions[end])"
    s *= "\n\n (†) id shown in-place of object"
    s *= "\n (‡) only last value listed"
    print(io, s)
end

"""
    Life{D} <: AbstractAgent

One of the two Agent types. Spawns from, travels to, and terraforms `Planet`s. Exists in space of dimension `D`.

See also [`Planet`](@ref)
"""
Base.@kwdef mutable struct Life{D} <:AbstractAgent
    id::Int
    pos::SVector{D, Float64}
    vel::SVector{D, Float64}
    parentplanet::Planet
    composition::Vector{<:Real} ## Taken from parentplanet
    destination::Planet
    destination_distance::Real
    ancestors::Vector{Life} ## Life agents that phylogenetically preceded this one
end
function Base.show(io::IO, life::Life{D}) where {D}
    s = "Life 🦠 in $(D)D space with properties:."
    s *= "\n id: $(life.id)"
    s *= "\n pos: $(life.pos)"
    s *= "\n vel: $(life.vel)"
    s *= "\n parentplanet (†): $(life.parentplanet.id)"
    s *= "\n composition: $(life.composition)"
    s *= "\n destination (†): $(life.destination.id)"
    s *= "\n destination_distance: $(life.destination_distance)"
    s *= "\n ancestors (†): $(length(life.ancestors) == 0 ? "No ancestors" : [i.id for i in life.ancestors])" ## Haven't tested the else condition here yet
    s *= "\n\n (†) id shown in-place of object"
    print(io, s)
end

"""
    random_positions(rng, maxdims::NTuple{D,X}, n) where {D,X<:Real}

Generate `n` random positions of dimension `D` within a tuple of maximum dimensions of the space given by `maxdim`.
"""
function random_positions(rng, maxdims::NTuple{D,X}, n) where {D,X<:Real}
    # Generate random values for each dimension
    random_vals = [rand(rng, Uniform(0, imax), n) for imax in maxdims]
    
    # Create SVectors from the random values
    result = Vector{SVector{D, Float64}}(undef, n)
    for i in 1:n
        result[i] = SVector{D, Float64}(getindex.(random_vals, i))
    end
    
    result
end

"""
    default_velocities(D,n) :: Vector{NTuple{D, Float64}}

Generate a vector of length `n` of `D` tuples, filled with 0s.
"""
function default_velocities(D, n)
    # Create a vector of SVectors filled with zeros
    return [SVector{D, Float64}(zeros(D)) for _ in 1:n]
end

"""
    random_compositions(rng, maxcomp, compsize, n)

Generate a `compsize` x `n` matrix of random integers between 1:`maxcomp`.
"""
random_compositions(rng, maxcomp, compsize, n) = rand(rng, Uniform(0,maxcomp), compsize, n)

"""
    random_radius(rng, rmin, rmax)

Generate a random radius between `rmin` and `rmax`.
"""
random_radius(rng, rmin, rmax) = sqrt(rand(rng) * (rmax^2 - rmin^2) + rmin^2)

"""
    filter_agents(model,agenttype)

Return only agents of type `agenttype` from `model`.
"""
filter_agents(model,agenttype) = Iterators.filter(a isa agenttype, allagents(model))

"""
    random_shell_position(rng, rmin, rmax)

Generate a random position within a spherical shell with thickness `rmax`-`rmin`.
"""
function random_shell_position(rng, rmin, rmax)
    valid_pos = false
    while valid_pos == false
        x,y,z = random_positions(rng, (rmax,rmax,rmax), 1)[1]
        (rmax > sqrt(x^2+y^2+z^2) > rmin) && (valid_pos = true)
        return x, y, z
    end
end


"""
    GalaxyParameters
    
Defines the AgentBasedModel, Space, and Galaxy

# Arguments
- `rng::AbstractRNG = Random.default_rng()`: the rng to pass to all functions.
- `extent::NTuple{D,<:Real} = (1.0, 1.0)`: the extent of the `Agents` space.
- `ABMkwargs::Union{Dict{Symbol},Nothing} = nothing`: kwargs to pass to `Agents.AgentBasedModel`.
- `SpaceArgs::Union{Dict{Symbol},Nothing} = nothing`: args to pass to `Agents.ContinuousSpace`.
- `SpaceKwargs::Union{Dict{Symbol},Nothing} = nothing`: kwargs to pass to `Agents.ContinuousSpace`.
- `dt::Real = 10`: `Planet`s move dt*vel every step; `Life` moves dt*lifespeed every step. 
- `lifespeed::Real = 0.2`: velocity of `Life`.
- `interaction_radius::Real = dt*lifespeed`: distance away at which `Life` can interact with a `Planet`.
- `ool::Union{Vector{Int}, Int, Nothing} = nothing`: id of `Planet`(s) on which to initialize `Life`.
- `nool::Int = 1`: the number of planets that are initialized with life
- `spawn_rate::Real = 0.02`: the frequency at which to send out life from every living planet (in units of dt) (only used for `galaxy_agent_step_spawn_at_rate!`)
- `compmix_func::Function = average_compositions`: Function to use for generating terraformed `Planet`'s composition. Must take as input two valid composition vectors, and return one valid composition vector.  
- `compmix_kwargs::Union{Dict{Symbol},Nothing} = nothing`: kwargs to pass to `compmix_func`.
- `compatibility_func::Function = compositionally_similar_planets`: Function to use for deciding what `Planet`s are compatible for future terraformation. 
- `compatibility_kwargs::Union{Dict{Symbol},Nothing} = nothing`: kwargs to pass to `compatibility_func`.
- `destination_func::Function = nearest_planet`:  Function to use for deciding which compatible `Planet` (which of the `planet.candidate_planet`s) should be the next destination. 
- `pos::Vector{<:NTuple{D,Real}}`: the initial positions of all `Planet`s.
- `vel::Vector{<:NTuple{D,Real}}`: the initial velocities of all `Planet`s.
- `maxcomp::Float64`: the max value of any element within the composition vectors.
- `compsize::Int`: the length of the compositon vectors.
- `planetcompositions::Array{Float64, 2}`: an array of default compositon vectors.

# Notes:
- `vel` defaults to 0 for all `Planet`s.
- `maxcomp` is used for any planets that are not specified when the model is initialized.
- `planetcompositions` are random for any planets that are not specified when the model is initialized.
- `compsize` must match any compositions provided.
...
"""
mutable struct GalaxyParameters
    rng # not part of ABMkwargs because it can previously be used for other things
    extent # not part of SpaceArgs because it can previously be used for other things
    ABMkwargs
    SpaceArgs
    SpaceKwargs
    dt
    lifespeed
    interaction_radius
    ool
    nool
    spawn_rate
    compmix_func
    compmix_kwargs
    compatibility_func
    compatibility_kwargs
    destination_func
    pos
    vel
    maxcomp
    compsize
    planetcompositions    

    function GalaxyParameters(;
        rng::AbstractRNG = Random.default_rng(),
        extent::NTuple{D,<:Real} = (1.0, 1.0), 
        ABMkwargs::Union{Dict{Symbol},Nothing} = nothing,
        SpaceArgs::Union{Dict{Symbol},Nothing} = nothing,
        SpaceKwargs::Union{Dict{Symbol},Nothing} = nothing,
        dt::Real = 10,
        lifespeed::Real = 0.2,
        interaction_radius::Real = dt*lifespeed,
        ool::Union{Vector{Int}, Int, Nothing} = nothing,
        nool::Int = 1,
        spawn_rate::Real = 0.02,
        compmix_func::Function = average_compositions,
        compmix_kwargs::Union{Dict{Symbol},Nothing} = nothing,
        compatibility_func::Function = compositionally_similar_planets,
        compatibility_kwargs::Union{Dict{Symbol},Nothing} = nothing,
        destination_func::Function = nearest_planet,
        pos::Vector{<:NTuple{D,Real}},
        vel::Vector{<:NTuple{D,Real}},
        maxcomp::Real,
        compsize::Int,
        planetcompositions::Array{<:Real, 2}) where {D}

        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        if ~all(x->length(x)==compsize, eachcol(planetcompositions))
            throw(ArgumentError("All planets compositions must have length of `compsize`"))
        end
        
        ## ABMkwargs
        if ABMkwargs === nothing 
            ABMkwargs = Dict(:rng => rng, :warn => false)
        elseif :rng in ABMkwargs
            rng != ABMkwargs[:rng] && throw(ArgumentError("rng and ABMkwargs[:rng] do not match. ABMkwargs[:rng] will inherit from rng if ABMkwargs[:rng] not provided."))
        else
            ABMkwargs[:rng] = rng
        end

        ## SpaceArgs
        if SpaceArgs === nothing
            SpaceArgs = Dict(:extent => extent)
        elseif :extent in SpaceArgs
            extent != SpaceArgs[:extent] && throw(ArgumentError("extent and SpaceArgs[:extent] do not match. SpaceArgs[:extent] will inherit from extent if SpaceArgs[:extent] not provided."))
        else
            SpaceArgs[:extent] = extent
        end

        ## SpaceKwargs
        SpaceKwargs === nothing && (SpaceKwargs = Dict(:periodic => true))
        
        new(rng, extent, ABMkwargs, SpaceArgs, SpaceKwargs, dt, lifespeed, interaction_radius, ool, nool, spawn_rate, compmix_func, compmix_kwargs, compatibility_func, compatibility_kwargs, destination_func, pos, vel, maxcomp, compsize, planetcompositions)

    end
    
end


"""
    GalaxyParameters(rng::AbstractRNG;
        pos::Union{Vector{<:NTuple{D,Real}}, Nothing} = nothing,
        vel::Union{Vector{<:NTuple{D,Real}}, Nothing} = nothing,
        planetcompositions::Union{Array{<:Real,2}, Nothing} = nothing,
        kwargs...) where {D}

Can be called with only `rng` and one of `pos`, `vel` or `planetcompositions`, plus any number of optional kwargs.

# Notes:
Uses GalaxyParameters(rng::AbstractRNG, nplanets::Int; ...) constructor for other arguments
"""
function GalaxyParameters(rng::AbstractRNG;
    pos::Union{Vector{<:NTuple{D,Real}}, Nothing} = nothing,
    vel::Union{Vector{<:NTuple{D,Real}}, Nothing} = nothing,
    planetcompositions::Union{Array{<:Real,2}, Nothing} = nothing,
    kwargs...) where {D}

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

    GalaxyParameters(rng, nplanets; args...)
end
"""
    GalaxyParameters(nplanets::Int; kwargs...)

The simplist way to initialize. Can be called with only `nplanets`, plus any number of optional kwargs.

# Notes:
Uses GalaxyParameters(rng::AbstractRNG, nplanets::Int; ...) constructor for other arguments
"""
function GalaxyParameters(nplanets::Int; kwargs...)
    GalaxyParameters(Random.default_rng(), nplanets; kwargs...) ## If it's ", kwargs..." instead of "; kwargs...", then I get an error from running something like this: TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0))
end

## Main external constructor for GalaxyParameters (other external constructors call it)
## Create with random pos, vel, planetcompositions
## Properties of randomized planetcompositions can be changed with new fields maxcomp, compsize
"""
    GalaxyParameters(rng::AbstractRNG, nplanets::Int;
        extent=(1.0, 1.0), 
        maxcomp=10, 
        compsize=10,
        pos=random_positions(rng, extent, nplanets),
        vel=default_velocities(length(extent), nplanets),
        planetcompositions=random_compositions(rng, maxcomp, compsize, nplanets),
        kwargs...)

The main external constructor for `GalaxyParameters` (other external constructors call it). Sets default values for 
`extent`, `maxcomp`, `compsize`, `pos` (random), `vel` (0), `planetcompositions` (random). Allows any number of optional kwargs.

# Notes:
Calls the internal constructor.
"""
function GalaxyParameters(rng::AbstractRNG, nplanets::Int;
    extent=(1.0, 1.0), 
    maxcomp=10, 
    compsize=10,
    pos=random_positions(rng, extent, nplanets),
    vel=default_velocities(length(extent), nplanets),
    planetcompositions=random_compositions(rng, maxcomp, compsize, nplanets),
    kwargs...)

    ## Calls the internal constructor. I still don't understand how this works and passes the correct keyword arguments to the correct places
    GalaxyParameters(; rng=rng, extent=extent, pos, vel, maxcomp, compsize, planetcompositions, kwargs...)
end

# """
#     Other kwargs of ABM besides properties (which fall into GalaxyProperties above)
#     Do I need this or can I wrap it into some other kwargs so that I don't have to copy and paste default kwargs?
# """
# struct ABMkwargs
#     scheduler
#     rng
#     warn

#     function ABMkwargs(
#         scheduler = Schedulers.fastest
#         rng::AbstractRNG = Random.default_rng()
#         warn = false)

#         new(scheduler, rng, warn)

#     end
# end

# """
#     All get passed into ContinuousSpace
# """
# struct SpaceArgs
#     extent
#     spacing
#     update_vel!
#     periodic

#     function SpaceArgs(
#         extent::NTuple{2,<:Real}=(1.0, 1.0),
#         spacing = min(extent...) / 10.0;
#         update_vel! = defvel,
#         periodic = true)

#         new(extent, spacing, update_vel!, periodic)
#     end

# end
"""
    nplanets(params::GalaxyParameters)

Return the number of planets in `params`.
"""
nplanets(params::GalaxyParameters) = length(params.pos)

"""
    count_living_planets(model)

Returns the number of alive planets.
"""
count_living_planets(model) = count(a -> a isa Planet && a.alive, allagents(model))

"""

    center_position(pos::NTuple{D,Real}, extent::NTuple{D,Real}, m::Real) where {D}

Assuming that the provided position is for the original `extent` size (of extent./m = original_extent), 
return the equivilent position at the center of current `extent` (original_extent.*m).
"""
center_position(pos::NTuple{D,Real}, extent::NTuple{D,Real}, m::Real) where {D} = pos.+((extent.-(extent./m))./2) 

"""
    galaxy_model_setup(params::GalaxyParameters)

Set up the galaxy model (planets and life) according to `params`. 

Calls [`galaxy_planet_setup`](@ref) and [`galaxy_life_setup`](@ref).
"""
function galaxy_model_setup(params::GalaxyParameters)

    model = galaxy_planet_setup(params)
    model = galaxy_life_setup(model, params::GalaxyParameters)
    model

end
"""
    galaxy_model_setup(params::Dict)

Initializes the GalaxyParameters struct from the provided dict.
"""
function galaxy_model_setup(params::Dict)

    params = GalaxyParameters(
        params[:rng],
        params[:nplanets]; 
        filter(x -> first(x) ∉ [:rng, :nplanets], params)...)
    model = galaxy_planet_setup(params)
    model = galaxy_life_setup(model, params::GalaxyParameters)
    model

end

"""
    Custom scheduler the ensures newly added agents don't get activated on the step they're added.
    
    The default scheduler is `keys(allagents(model))` which gets modified in-place and causes problems.
"""
allocated_fastest(model::ABM) = collect(allids(model))

"""
    galaxy_planet_setup(params::GalaxyParameters)

Set up the galaxy's `Planet`s according to `params`.

Called by [`galaxy_model_setup`](@ref).
"""
function galaxy_planet_setup(params::GalaxyParameters)

    extent_multiplier = 1
    params.extent = extent_multiplier.*params.extent

    if :spacing in keys(params.SpaceArgs)
        space = ContinuousSpace(params.extent, params.SpaceArgs[:spacing]; params.SpaceKwargs...)
    else
        space = ContinuousSpace(params.extent; params.SpaceKwargs...)
    end

    model = @suppress_err AgentBasedModel(
        Union{Planet,Life},
        space,
        scheduler = allocated_fastest,
        properties = Dict(:dt => params.dt,
                        :lifespeed => params.lifespeed,
                        :interaction_radius => params.interaction_radius,
                        :nplanets => nplanets(params),
                        :maxcomp => params.maxcomp,
                        :compsize => params.compsize,
                        :s => 0, ## track the model step number,
                        :n_living_planets => params.nool,
                        :terraformed_on_step => true,
                        :n_terraformed_on_step => params.nool,
                        :spawn_rate => params.spawn_rate,
                        :GalaxyParameters => params,
                        :compmix_func => params.compmix_func,
                        :compmix_kwargs => params.compmix_kwargs,
                        :compatibility_func => params.compatibility_func,
                        :compatibility_kwargs => params.compatibility_kwargs,
                        :destination_func => params.destination_func);
                        # :nlife => length(params.ool)
                        # :ool => params.ool,
                        # :pos => params.pos,
                        # :vel => params.vel,
                        # :planetcompositions => params.planetcompositions); ## Why does having a semicolon here fix it???
        # rng=params.ABMkwargs[:rng],
        # warn=params.ABMkwargs[:warn]
        params.ABMkwargs... ## Why does this not work??
    )

    initialize_planets!(model, params, extent_multiplier)
    model

end

"""
    galaxy_life_setup(params::GalaxyParameters)

Set up the galaxy's `Planet`s according to `params`.

Called by [`galaxy_model_setup`](@ref).
"""
function galaxy_life_setup(model, params::GalaxyParameters)

    ## Initialize living planets
    for _ in 1:params.nool

        planet = 
            isnothing(params.ool) ? random_agent(model, x -> x isa Planet && !x.alive && !x.claimed ) : model[params.ool]
        
        planet.alive = true
        planet.claimed = true

    end

    ## Spawn life (candidate planets have to be calculated after all alive planets are initialized)
    for planet in Iterators.filter(a -> a isa Planet && a.alive, allagents(model))

        # planet.candidate_planets = compatibleplanets(planet, model)
        find_compatible_planets!(planet, model)
        spawn_if_candidate_planets!(planet, model)

    end

    model

end

"""
    initialize_planets!(model, params::GalaxyParameters, extent_multiplier)

Initialize `Planet`s in the galaxy.

`Planet` positions are adjusted to `center_position`, based on `extent_multiplier`.

This acts to increase the space seen by the user when plotting, and put the simulation in the center of the space, 
so that there is room to add more planets.

Called by [`galaxy_model_setup`](@ref).
"""
function initialize_planets!(model, params::GalaxyParameters, extent_multiplier)
    for i = 1:nplanets(params)
        id = Agents.nextid(model)
        pos = center_position(params.pos[i], params.extent, extent_multiplier)
        vel = params.vel[i]
        composition = params.planetcompositions[:, i]

        planet = Planet(; id=id, pos=SA[pos...], vel=SA[vel...], composition=composition)

        add_agent_own_pos!(planet, model)
    end
    model
end

"""
    basic_candidate_planets(planet::Planet, model::ABM)

Returns possible candidate planets filtered by the most basic requirements.

e.g. Destination Planet not alive, claimed, or the same Planet as the parent.
"""
function basic_candidate_planets(planet::Planet, model::ABM)
    candidates = Iterators.filter(p -> 
        p isa Planet && 
        !p.alive && 
        !p.claimed && 
        p.id != planet.id, 
        allagents(model)
    )
    
    return convert(Vector{Planet}, collect(candidates))
end

function planet_attribute_as_matrix(planets::Vector{Planet}, attr::Symbol)

    length(planets) == 0 && throw(ArgumentError("planets is empty"))
    planet_attributes = map(x -> getproperty(x, attr), planets)
    ## need to collect because when attr = :pos, the result is a Vector of Tuples
    convert(Matrix{Float64}, hcat(collect.(planet_attributes)...))
    # NOTE: I hope it doesn't cause problems that the returned matrix has element type of whatever the attribute is
    # lol it already is. converting to float matrix.

end

"""
    compositionally_similar_planets(planet::Planet, model::ABM, allowed_diff::Real = 1.0)

Return `Vector{Planet}` of `Planet`s compatible with `planet` for terraformation, based on compositional similarity.

A valid `compatibility_func`.
"""
function compositionally_similar_planets(planet::Planet, model::ABM; allowed_diff)
## Alternative name: planets_in_composition_range(planet::Planet, model::ABM; allowed_diff)
    candidateplanets = basic_candidate_planets(planet, model)
    length(candidateplanets)==0 && return Vector{Planet}[]
    planets_in_attribute_range(planet, candidateplanets, :composition, allowed_diff)
    # convert(Vector{Planet}, candidateplanets[compatibleindxs]) ## Returns Planets. Needed in case the result is empty? But it should still return an empty Vector{Planet} I think
end


"""
    nearest_k_planets(planet::Planet, planets::Vector{PLanet}, k)

Returns nearest `k` planets

A valid `compatibility_func`.

Note: Results are unsorted
"""
function nearest_k_planets(planet::Planet, planets::Vector{Planet}, k)
    
    planetpositions = planet_attribute_as_matrix(planets, :pos)
    idxs, dists = knn(KDTree(planetpositions), Vector(planet.pos), k)
    planets[idxs]

end
function nearest_k_planets(planet::Planet, model::ABM; k)
    
    candidateplanets = basic_candidate_planets(planet, model)

    n_candidateplanets = length(candidateplanets)
    if n_candidateplanets==0
        return Vector{Planet}[]
    elseif k > n_candidateplanets
        k = n_candidateplanets
    end

    nearest_k_planets(planet, candidateplanets, k)

end

"""
    planets_in_attribute_range(planet::Planet, planets::Vector{Planet}, attr::Symbol, r)

Returns all planets within range `r` of the `attr` space (unsorted).

Used for `compatibility_func`s.

Called by [`planets_in_range`](@ref).
"""
function planets_in_attribute_range(planet::Planet, planets::Vector{Planet}, attr::Symbol, r)

    planetattributes = planet_attribute_as_matrix(planets, attr)
    idxs = inrange(KDTree(planetattributes), Vector(getproperty(planet, attr)), r)
    planets[idxs]

end

planets_in_range(planet::Planet, planets::Vector{Planet}, r) = planets_in_attribute_range(planet, planets, :pos, r)
function planets_in_range(planet::Planet, model::ABM; r)

    candidateplanets = basic_candidate_planets(planet, model)
    length(candidateplanets)==0 && return Vector{Planet}[]
    planets_in_range(planet, candidateplanets, r)

end


"""
    closest_planet_by_attribute(planet::Planet, planets::Vector{Planet}, attr::Symbol)

Returns `Planet` within `planets` that is most similar in `attr` space.

Used for `destination_func`s.

Called by [`nearest_planet`](@ref), [`most_similar_planet`](@ref).
"""
function closest_planet_by_attribute(planet::Planet, planets::Vector{Planet}, attr::Symbol)

    planetattributes = planet_attribute_as_matrix(planets, attr)
    idx, dist = nn(KDTree(planetattributes), Vector(getproperty(planet, attr)))
    planets[idx]

end

nearest_planet(planet::Planet, planets::Vector{Planet}) = closest_planet_by_attribute(planet, planets, :pos)
most_similar_planet(planet::Planet, planets::Vector{Planet}) = closest_planet_by_attribute(planet, planets, :composition)

function spawn_if_candidate_planets!(
    planet::Planet,
    model::ABM,
    life::Union{Life,Nothing} = nothing
)
    ## Only spawn life if there are compatible Planets
    candidateplanets = planet.candidate_planets
    # if length(candidateplanets) == 0
    #     println("Planet $(planet.id) has no compatible planets. It's the end of its line.")
    if length(candidateplanets) != 0
        isnothing(life) ?  spawnlife!(planet, model) : spawnlife!(planet, model, ancestors = push!(life.ancestors, life))
    end
    model
end

"""
    spawnlife!(planet::Planet, model::ABM; ancestors::Vector{Life} = Life[])

Spawns `Life` at `planet`.

Called by [`galaxy_model_setup`](@ref) and [`terraform!`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM;
    ancestors::Vector{Life} = Life[]
    )

    destinationplanet = model.destination_func(planet, planet.candidate_planets) #model.compatibility_func(planet, model) #nearest_planet(planet, planet.candidate_planets)
    destination_distance = distance(destinationplanet.pos, planet.pos)
    vel = direction(planet, destinationplanet) .* model.lifespeed

    life = Life(;
        id = Agents.nextid(model),
        pos = planet.pos,
        vel = SA[vel...],
        parentplanet = planet,
        composition = planet.composition,
        destination = destinationplanet,
        destination_distance = destination_distance,
        ancestors
    ) ## Only "first" life won't have ancestors

    life = add_agent_own_pos!(life, model)

    destinationplanet.claimed = true 
    ## NEED TO MAKE SURE THAT THE FIRST LIFE HAS PROPERTIES RECORDED ON THE FIRST PLANET
    model

end

"""
    average_compositions(lifecomposition::Vector{Float64}, planetcomposition::Vector{Float64})

Default composition mixing function (`compmix_func`). Rounds element-averaged composition between two compositon vectors.

Can be overridden by providing a custom `compmix_func` when setting up `GalaxyParameters`.

Custom function to use for generating terraformed `Planet`'s composition must likewise take as input two valid composition 
vectors, and return one valid composition vector.  

`model::ABM` is a required param in order to have a standardize argumnet list for all `compmix_func`s

See [`GalaxyParameters`](@ref).

Related: [`crossover_one_point`](@ref).
"""
function average_compositions(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, model::ABM)
    ## Simple for now; Rounding goes to nearest even number
    average_compositions(lifecomposition, planetcomposition)
end

"""
    average_compositions(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real})

Can be called without `model::ABM` arg.
"""
function average_compositions(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real})
    ## Simple for now; Rounding goes to nearest even number
    round.((lifecomposition .+ planetcomposition) ./ 2)
end

"""
    crossover_one_point(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, model::ABM, mutation_rate=1/length(lifecomposition))

A valid `compmix_func`. Performs one-point crossover between the `lifecomposition` and `planetcomposition`.

The crossover point is after the `crossover_after_idx`, which is limited between 1 and length(`lifecomposition`)-1).

The returned strand and crossover point are randomly chosen based on `abmrng(model)``.

If mutated, substitute elements are chosen from random distribution of `Uniform(0, model.maxcomp)`.

See: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)

Related: [`average_compositions`](@ref), [`mutate_strand`](@ref), [`positions_to_mutate`](@ref).
"""
function crossover_one_point(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, model::ABM; mutation_rate=1/length(lifecomposition))
    crossover_one_point(lifecomposition, planetcomposition, abmrng(model); mutation_rate, model.maxcomp)
end
"""
    crossover_one_point(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, crossover_after_idx::Int)

Deterministic variant that requires specifying the `crossover_after_idx`, and returns both strands.
"""
function crossover_one_point(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, crossover_after_idx::Int)

    strand_1 = vcat(lifecomposition[1:crossover_after_idx], planetcomposition[crossover_after_idx+1:end])
    strand_2 = vcat(planetcomposition[1:crossover_after_idx], lifecomposition[crossover_after_idx+1:end])

    return strand_1, strand_2

end
"""
    crossover_one_point(
        lifecomposition::Vector{<:Real}, 
        planetcomposition::Vector{<:Real}, 
        rng::AbstractRNG = Random.default_rng(),
        mutation_rate=1/length(lifecomposition), 
        maxcomp=1)

Can be called with an rng object directly.
"""
function crossover_one_point(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, rng::AbstractRNG = Random.default_rng(); mutation_rate=1/length(lifecomposition), maxcomp=1)
    ## choose random index to start crossover, making sure that both strands contain 
    ##  at least 1 element from each parent composition
    crossover_after_idx = rand(rng, 1:length(lifecomposition)-1)
    ## coin flip to decide if we keep idx:end of parent1 or parent2
    strand_1, strand_2 = crossover_one_point(lifecomposition, planetcomposition, crossover_after_idx)
    ## return one of the two strands
    return_strand = rand(rng, 0:1) == 0 ? strand_1 : strand_2
    
    return mutate_strand(return_strand, maxcomp, rng, mutation_rate)
end

# function horizontal_gene_transfer(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, model::ABM; mutation_rate=1/length(lifecomposition))
#     horizontal_gene_transfer(lifecomposition, planetcomposition, abmrng(model); mutation_rate, model.maxcomp)
# end
function horizontal_gene_transfer(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, model::ABM; mutation_rate=1/length(lifecomposition), n_idxs_to_keep_from_destination=1)
    horizontal_gene_transfer(lifecomposition, planetcomposition, abmrng(model); mutation_rate, model.maxcomp, n_idxs_to_keep_from_destination)
end

function horizontal_gene_transfer(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, rng::AbstractRNG = Random.default_rng(); mutation_rate=1/length(lifecomposition), maxcomp=1, n_idxs_to_keep_from_destination=1)
    # new_strand = Array{typeof(lifecomposition[1])}(undef, length(lifecomposition))
    idxs_to_keep_from_destination = StatsBase.sample(rng, 1:length(planetcomposition), n_idxs_to_keep_from_destination, replace=false)
    new_strand = horizontal_gene_transfer(lifecomposition, planetcomposition, idxs_to_keep_from_destination)
    return mutate_strand(new_strand, maxcomp, rng, mutation_rate)
end

function horizontal_gene_transfer(lifecomposition::Vector{<:Real}, planetcomposition::Vector{<:Real}, idxs_to_keep_from_destination::Vector{Int})
    new_strand = deepcopy(lifecomposition)
    for i in idxs_to_keep_from_destination
        new_strand[i] = copy(planetcomposition[i])
    end
    new_strand
end


"""
    mutate_strand(strand::Vector{<:Real}, maxcomp, rng::AbstractRNG = Random.default_rng(), mutation_rate=1/length(strand))

Mutates elements in provided strand with probability of `mutation_rate`. 

Substituted elements are chosen from random distribution of `Uniform(0, model.maxcomp)`.

Related: [`crossover_one_point`](@ref).
"""
function mutate_strand(strand::Vector{<:Real}, maxcomp, rng::AbstractRNG = Random.default_rng(), mutation_rate=1/length(strand))
    random_strand = rand(rng, length(strand))
    position_strand = positions_to_mutate(random_strand, mutation_rate)    
    mutated_values = rand(rng, Uniform(0,maxcomp), sum(position_strand))
    strand[position_strand.==1] .= mutated_values ## will all the random values be different here? need to test
    return strand
end

"""
    positions_to_mutate(random_strand, mutation_rate=1/length(random_strand))

Returns a vector of 0s and 1s, where 1s indicate positions that will be mutated by `mutate_strand`.

Related: [`mutate_strand`](@ref).
"""
positions_to_mutate(random_strand, mutation_rate=1/length(random_strand)) = random_strand .< (ones(length(random_strand)) .* mutation_rate)

## Should I make the below funciton take as input (life.composition, planet.composition, model) instead of (life, planet, model)?
mix_compositions!(life::Life, planet::Planet, model::ABM) = isnothing(model.compmix_kwargs) ? planet.composition = model.compmix_func(life.composition, planet.composition, model) : planet.composition = model.compmix_func(life.composition, planet.composition, model; model.compmix_kwargs...)

## I think I can actually simplify the below since all compatible_planet functions are going to take the model as input, so I don't need to check for the precense of model.compatibility_kwargs first? maybe?
find_compatible_planets!(planet::Planet, model::ABM) = isnothing(model.compatibility_kwargs) ? planet.candidate_planets = model.compatibility_func(planet, model) : planet.candidate_planets = model.compatibility_func(planet, model; model.compatibility_kwargs...)

"""
    terraform!(life::Life, planet::Planet, model::ABM)

Performs actions on `life` and `planet` associated with successful terraformation. Takes
existing `life` and terraforms an exsiting non-alive `planet` (not user facing).
- Mix the `composition` of `planet` and `life`
- Update the `planet` to `alive=true`
- Update the `planet`'s `ancestors`, `parentplanet`, `parentlife`, and `parentcomposition`
- Call `spawnlife!` to send out `Life` from `planet`.

Called by [`galaxy_agent_step_spawn_on_terraform!`](@ref).
"""
function terraform!(life::Life, planet::Planet, model::ABM)

    ## Modify destination planet properties
    mix_compositions!(life, planet, model)
    planet.alive = true
    push!(planet.parentlifes, life)
    push!(planet.parentplanets, life.parentplanet)
    push!(planet.parentcompositions, life.composition)
    
    ## Calculate candidate planets
    find_compatible_planets!(planet, model)

    # planet.claimed = true ## Test to make sure this is already true beforehand
end

"""
    pos_is_inside_alive_radius(pos::Tuple, model::ABM)

Return `false` if provided `pos` lies within any life's interaction radii    
"""
function pos_is_inside_alive_radius(pos::Tuple, model::ABM, exact=true)

    exact==true ? neighbor_func = nearby_ids_exact : nearby_ids
    
    neighbor_ids = collect(neighbor_func(pos,model,model.interaction_radius))

    if count(a -> a.id in neighbor_ids && a isa Planet && a.alive, allagents(model)) > 0
        return true
    else
        return false
    end

end

"""
    galaxy_model_step!(model)

Custom `model_step` to be called by `Agents.step!`. 

# Notes:
Right now this only updates the number of planets in the simulation if the interactive slider is changed.
"""
function galaxy_model_step!(model)
    
    current_n_living_planets = count_living_planets(model)

    if current_n_living_planets > model.n_living_planets
        model.terraformed_on_step = true
        model.n_terraformed_on_step = current_n_living_planets - model.n_living_planets
        model.n_living_planets = current_n_living_planets
    else 
        model.n_terraformed_on_step = 0
        model.terraformed_on_step = false
    end

    # model.n_terraformed_on_step = max_life_id(model) - model.max_life_id
    model.s += 1

end

"""
    galaxy_agent_step_spawn_on_terraform!(life::Life, model)

Custom `agent_step!` for Life. 

    - Moves `life`
    - If `life` is within 1 step of destination planet, `terraform!`s life's destination, and kills `life`.

Avoids using `Agents.nearby_ids` because of bug (see: https://github.com/JuliaDynamics/Agents.jl/issues/684).
"""
function galaxy_agent_step_spawn_on_terraform!(life::Life, model)

    if life.destination == nothing
        remove_agent!(life, model)
    elseif life.destination_distance < model.dt*hypot((life.vel)...)
        terraform!(life, life.destination, model)
        spawn_if_candidate_planets!(life.destination, model, life)
        remove_agent!(life, model)
    else
        move_agent!(life, model, model.dt)
        life.destination_distance = distance(life.pos, life.destination.pos)
    
    end

end

"""
    galaxy_agent_step_spawn_on_terraform!(planet::Planet, model)

Custom `agent_step!` for Planet. Doesn't do anything. Only needed because we have an `agent_step!`
function for `Life`.
"""
function galaxy_agent_step_spawn_on_terraform!(planet::Planet, model)

    ## Don't need to update candidate planets for planets which are already alive if the spawn is only on terraform
    dummystep(planet, model)

end

"""
    galaxy_agent_step_spawn_at_rate!(life::Life, model)

Custom `agent_step!` for Life. 

    - Moves `life`
    - If `life` is within 1 step of destination planet, `terraform!`s life's destination, and kills `life`.

Avoids using `Agents.nearby_ids` because of bug (see: https://github.com/JuliaDynamics/Agents.jl/issues/684).
"""
function galaxy_agent_step_spawn_at_rate!(life::Life, model)

    if life.destination == nothing
        remove_agent!(life, model)
    elseif life.destination_distance < model.dt*hypot((life.vel)...)
        terraform!(life, life.destination, model)
        remove_agent!(life, model)
    else
        move_agent!(life, model, model.dt)
        life.destination_distance = distance(life.pos, life.destination.pos)
    end

end

"""
    galaxy_agent_step_spawn_at_rate!(planet::Planet, model)

Custom `agent_step!` for Planet. Spawns life at a fixed rate. 
"""
function galaxy_agent_step_spawn_at_rate!(planet::Planet, model)

    planet.alive && (planet.spawn_threshold += model.dt * model.spawn_rate)

    if planet.spawn_threshold >= 1

        ## update candidate planets 
        filter!(p-> !p.alive && !p.claimed, planet.candidate_planets)
        length(planet.parentlifes) > 0 ? life = planet.parentlifes[end] : life = nothing
        spawn_if_candidate_planets!(planet, model, life)
        planet.spawn_threshold = 0

    end

end



function galaxy_agent_direct_step!(life::Life, model)
    
    if life.destination == nothing
        remove_agent!(life, model)
    elseif isapprox(life.destination_distance, 0, atol=0.5)
        terraform!(life, life.destination, model)
        remove_agent!(life, model)
    else
        move_agent!(life, life.destination.pos, model)
        life.destination_distance = distance(life.pos, life.destination.pos)
    end

end

function galaxy_agent_direct_step!(planet::Planet, model)

    # move_agent!(planet, model, model.dt)
    dummystep(planet, model)

end



#######################################
## Distances, Correlations and permutations
"""

    concatenate_planet_fields(field, model, planet_condition=nothing)

Concatenate all the of a model's planet's fields into a single m x n matrix.

Planets are filtered by the provided optional `planet_condition` arg.

e.g. Concatonate the :composition of every planet into a single matrix.

"""
function concatenate_planet_fields(field, model, planet_condition=nothing)

    field_values = []
    for (id,planet) in filter_agents(model,Planet)

        if planet_condition != nothing             
            planet_condition(planet) && push!(field_values, collect(getfield(planet, field)))
        else
            push!(field_values, collect(getfield(planet, field)))
        end

    end

    hcat(field_values...)

end

function PermuteDistanceMatrix(D; rng::AbstractRNG = Random.default_rng())
    order = shuffle(rng, collect(1:size(D)[1]))
    return D[order,:][:,order]
end

triangleFlatten(D) = D[tril!(trues(size(D)), -1)]

"""

    PlanetMantelTest(model, xfield=:composition, yfield=:pos, dist_metric=Euclidean();  method=:pearson, permutations=999, alternative=:twosided, planet_condition=nothing)

xfield::symbol is meant to be one of (:composition, :pos)
yfield::symbol is meant to be one of (:composition, :pos)
dist_metric::dist from Distances.jl (default: Euclidean())
planet_condition takes an optional function which can be used to filter planets. For example, filtering by only planets which are alive. 
"""
function PlanetMantelTest(model, xfield=:composition, yfield=:pos; dist_metric=Euclidean(),  method=:pearson, permutations=999, alternative=:twosided, planet_condition=nothing)

    x = pairwise(dist_metric, concatenate_planet_fields(xfield, model, planet_condition), dims=2)
    y = pairwise(dist_metric, concatenate_planet_fields(yfield, model, planet_condition), dims=2)

    MantelTest(x, y;  rng=abmrng(model), dist_metric=dist_metric, method=method, permutations=permutations, alternative=alternative)

end

function MantelTest(x, y;  rng::AbstractRNG = Random.default_rng(), dist_metric=Euclidean(), method=:pearson, permutations=999, alternative=:twosided)
    ## Based this off of https://github.com/biocore/scikit-bio/blob/0.1.3/skbio/math/stats/distance/_mantel.py#L23

    method == :pearson ? corr_func = cor : throw(ArgumentError("Not yet implemented")) 

    permutations < 0 && throw(ArgumentError("Number of permutations must be greater than or equal to zero."))
    alternative ∉ [:twosided, :greater, :less] && throw(ArgumentError("Invalid alternative hypothesis $alternative."))

    ## Below not needed because x and y must already be distance matrices here
    # x = pairwise(dist_metric, x, dims=2)
    # y = pairwise(dist_metric, y, dims=2)

    ## Check properties to verify they're distance matrices
    size(x) != size(y) && throw(ArgumentError("Distance matrices must have the same shape."))
    size(x)[1] < 3 && throw(ArgumentError("Distance matrices must be at least 3x3 in size."))
    sum(abs.(diag(x))) + sum(abs.(diag(y))) != 0 && throw(ArgumentError("Distance matrices must be hollow. I think."))
    if ~issymmetric(x)
        show(stdout, "text/plain", x)
        @show x
        any(isnan, x) && @show findall(isnan, x)
    end
    if ~issymmetric(y)
        show(stdout, "text/plain", y)
        @show y 
        any(isnan, y) && @show findall(isnan, y)
    end
    ~issymmetric(x) | ~issymmetric(y) && throw(ArgumentError("Distance matrices must be symmetric. I think."))

    ## This part just needs to get a flattened version of the diagonal of a hollow, square, symmetric matrix
    x_flat = triangleFlatten(x)
    y_flat = triangleFlatten(y)

    orig_stat = corr_func(x_flat, y_flat)

    ## Permutation tests
    if (permutations == 0) | isnan(orig_stat)
        p_value = NaN
    else
        perm_gen = (cor(triangleFlatten(PermuteDistanceMatrix(x,rng=rng)), y_flat) for _ in 1:permutations)
        permuted_stats = collect(Iterators.flatten(perm_gen))

        if alternative == :twosided
            count_better = sum(abs.(permuted_stats) .>= abs(orig_stat))
        elseif alternative == :greater
            count_better = sum(permuted_stats .>= orig_stat)
        else
            count_better = sum(permuted_stats .<= orig_stat)
        end

        p_value = (count_better + 1) / (permutations + 1)
        
    end

    return orig_stat, p_value

end

##############################################################################################################################
## Data collection utilities 
##############################################################################################################################
## Some functions to nest and unnest data for csv saving
function unnest_agents(df_planets, ndims, compsize)
    # inspired from https://bkamins.github.io/julialang/2022/03/11/unnesting.html
    # TODO streamline naming
    df = transform(df_planets, :pos => AsTable)
    
    if ndims == 1
        new_names = Dict("x1" => "x")
    elseif ndims == 2
        new_names = Dict("x1" => "x", "x2" => "y")
    elseif ndims == 3
        new_names = Dict("x1" => "x", "x2" => "y", "x3" => "z")
    end
    rename!(df, new_names)

    ## Don't even save velocity data anymore
    # df = transform(df, :vel => AsTable)
    # if ndims == 1
    #     new_names = Dict("x1" => "v_x")
    # elseif ndims == 2
    #     new_names = Dict("x1" => "v_x", "x2" => "v_y")
    # elseif ndims == 3
    #     new_names = Dict("x1" => "v_x", "x2" => "v_y", "x3" => "v_z")
    # end
    # rename!(df, new_names)

    df = transform(df, :composition => AsTable)
    new_names = Dict("x$i" => "comp_$(i)" for i in 1:compsize)
    rename!(df, new_names)

    select!(df, Not([:pos, :composition, :agent_type]))
    return df
end


function unnest_planets(df_planets, ndims, compsize)
    df = unnest_agents(df_planets, ndims, compsize)

    ## Don't write initial composition anymore
    # df = transform(df, :initialcomposition => AsTable)
    # new_names = Dict("x$i" => "init_comp_$(i)" for i in 1:compsize)
    # rename!(df, new_names)
    
    # select!(df, Not([:initialcomposition]))
    return df
end

function unnest_life(df, ndims, compsize)
    df = transform(df, :composition => AsTable)
    new_names = Dict("x$i" => "comp_$(i)" for i in 1:compsize)
    rename!(df, new_names)

    select!(df, Not([:composition, :agent_type]))
    return df
end

function split_df_agent(df_agent, model)
    split_df_agent(agent, length(model.space.dims), model.properties[:compsize])
    return df_planets, df_lifes
end

function split_df_agent(df_agent, dims, compsize)
    df_planets = df_agent[.! ismissing.(df_agent.alive),:]
    select!(df_planets, Not([:destination_distance,
                            :destination_id, :ancestor_ids])) # also: 
    df_planets = unnest_planets(df_planets, dims, compsize)

    # misspell for parsing reasons
    df_lifes = df_agent[ismissing.(df_agent.alive),:]
    select!(df_lifes, Not([:alive, :claimed, :pos])) #parentplanets ids??
    df_lifes = unnest_life(df_lifes, dims, compsize)

    return df_planets, df_lifes
end

function clean_df(df)
    ## Replace nothings with missings
    mapcols!(col -> replace(col, nothing => missing), df)
    ## Recast the column types (makes column go from e.g. Union{Bool,Missing} -> Bool)
    df = identity.(df)
    ## Replace true/false Bool columns with 0s and 1s
    mapcols!(col -> eltype(col) == Bool ? Int8.(col) : col, df)

    return df 
end


# rng = MersenneTwister(3141)
# x = [[0,1,2],[1,0,3],[2,3,0]]
# y = [[0, 2, 7],[2, 0, 6],[7, 6, 0]]
# MantelTest(hcat(x...),hcat(y...), rng=rng)

## Fun with colors
# col_to_hex(col) = "#"*hex(col)
# hex_to_col(hex) = convert(RGB{Float64}, parse(Colorant, hex))
# mix_cols(c1, c2) = RGB{Float64}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)

##############################################################################################################################
## Interactive Plot utilities 
##     REMOVED DUE TO ISSUE WITH REQUIRES, SEE: https://github.com/JuliaPackaging/Requires.jl/issues/111
##     UPDATE 2023-05-29: I made a work around; Now in the script you just have to add e.g. InteractiveDynamics.agent2string(agent::Life) = TerraformingAgents.agent2string(agent::Life)
##############################################################################################################################
"""
Overload InteractiveDynamics.jl's agent2string function in order to force interactive plot hover text to display only 
information for the ids under the cursor (instead of including nearby ids)

For more information see: 
https://github.com/JuliaDynamics/InteractiveDynamics.jl/blob/4a701abdb40abefc9e3bc6161bb223d22cd2ef2d/src/agents/inspection.jl#L99
"""
function agent2string(model::Agents.ABM{<:ContinuousSpace}, agent_pos)
    ids = Agents.nearby_ids_exact(agent_pos, model, 0.0)

    s = ""

    for id in ids
        s *= agent2string(model[id]) * "\n"
    end

    return s
end

"""
Overload InteractiveDynamics.jl's agent2string function with custom fields for Planets

For more information see: https://juliadynamics.github.io/InteractiveDynamics.jl/dev/agents/#InteractiveDynamics.agent2string
https://stackoverflow.com/questions/37031133/how-do-you-format-a-string-when-interpolated-in-julia
"""
function agent2string(agent::Planet)
    """
    ✨ Planet ✨
    id = $(agent.id)
    pos = ($(join([@sprintf("%.2f", i) for i in agent.pos],", ")))
    vel = $(agent.vel)
    composition = [$(join([@sprintf("%.2f", i) for i in agent.composition],", "))]
    initialcomposition = [$(join([@sprintf("%.2f", i) for i in agent.initialcomposition],", "))]
    alive = $(agent.alive)
    claimed = $(agent.claimed)
    parentplanets (†‡): $(length(agent.parentplanets) == 0 ? "No parentplanet" : agent.parentplanets[end].id)
    parentlifes (†‡): $(length(agent.parentlifes) == 0 ? "No parentlife" : agent.parentlifes[end].id)
    parentcompositions (‡): $(length(agent.parentcompositions) == 0 ? "No parentcomposition" : "[$(join([@sprintf("%.2f", i) for i in agent.parentcompositions[end]],", "))]")
    """
    ## Have to exclude this because it's taking up making the rest of the screen invisible
    # ancestor_ids = $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    
end

"""
Overload InteractiveDynamics.jl's agent2string function with custom fields for Life

For more information see: https://juliadynamics.github.io/InteractiveDynamics.jl/dev/agents/#InteractiveDynamics.agent2string
https://stackoverflow.com/questions/37031133/how-do-you-format-a-string-when-interpolated-in-julia
"""
function agent2string(agent::Life)
    """
    ✨ Life ✨
    id = $(agent.id)
    pos = ($(join([@sprintf("%.2f", i) for i in agent.pos],", ")))
    vel = ($(join([@sprintf("%.2f", i) for i in agent.vel],", ")))
    parentplanet (†): $(agent.parentplanet.id)
    composition = [$(join([@sprintf("%.2f", i) for i in agent.composition],", "))]
    destination (†): $(agent.destination.id)
    destination_distance: $(agent.destination_distance)
    ancestors (†): $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    """
    ## Have to exclude this because it's taking up making the rest of the screen invisible
    # ancestor_ids = $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    
end

end # module

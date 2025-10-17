module TerraformingAgents;

# include("utilities.jl")

using Agents, Random, Printf
using Statistics: cor
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot, diag, issymmetric, norm, tril!
using Distributions: Uniform, Normal
using NearestNeighbors
using Distances
using DataFrames
using StatsBase
using StaticArrays
using CSV

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
       clean_df,
       count_living_planets,
       nplanets

export NBodyData, load_nbody_data, get_nbody_position,
       calculate_reachable_destinations_nbody, spawn_if_candidate_planets_nbody!,
       spawnlife_mission!, galaxy_agent_step_nbody!, galaxy_model_step_nbody!

export CNS5Data, load_cns5_catalog

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
    speed(a)

Return speed of Agent based on its velocity, rounded to 10 digits.
"""
speed(a::AbstractAgent) = round(hypot(a.vel...), digits=10)

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

    reached_boundary::Bool = false

    # New fields for N-body integration (with defaults for backward compatibility)
    last_launch_timestep::Int = -1000  # When planet last launched a mission
end
function Base.show(io::IO, planet::Planet{D}) where {D}
    s = "Planet in $(D)D space with properties:."
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
    s *= "\n last_launch_timestep: $(planet.last_launch_timestep)"
    s *= "\n reached boundary: $(planet.reached_boundary)"
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
    # New fields for N-body integration (with defaults for backward compatibility)
    departure_timestep::Int = -1  # When mission was launched (-1 = not a mission)
    arrival_timestep::Int = -1    # When mission will arrive (-1 = not a mission)  
    is_mission::Bool = false      # True if this is a teleporting mission
end
function Base.show(io::IO, life::Life{D}) where {D}
    s = "Life in $(D)D space with properties:."
    s *= "\n id: $(life.id)"
    s *= "\n pos: $(life.pos)"
    s *= "\n vel: $(life.vel)"
    s *= "\n parentplanet (†): $(life.parentplanet.id)"
    s *= "\n composition: $(life.composition)"
    s *= "\n destination (†): $(life.destination.id)"
    s *= "\n destination_distance: $(life.destination_distance)"
    s *= "\n ancestors (†): $(length(life.ancestors) == 0 ? "No ancestors" : [i.id for i in life.ancestors])" ## Haven't tested the else condition here yet
    s *= "\n departure_timestep: $(life.departure_timestep)"
    s *= "\n arrival_timestep: $(life.arrival_timestep)"
    s *= "\n is_mission: $(life.is_mission)"
    s *= "\n\n (†) id shown in-place of object"
    print(io, s)
end

get_ancestors(life::Union{Life,Nothing}) = isnothing(life) ? Life[] : push!(copy(life.ancestors), life)

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
filter_agents(model,agenttype) = Iterators.filter(a->a isa agenttype, allagents(model))

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
Data structure to hold N-body trajectory data that integrates with existing GalaxyParameters
"""
struct NBodyData
    # Core position data indexed by (star_id, timestep) 
    positions::Dict{Tuple{Int,Int}, SVector{3,Float64}}
    
    # Time information extracted from the data
    timesteps::Vector{Int}
    times::Vector{Float64}  # Actual time values from CSV
    star_ids::Vector{Int}
    dt::Float64  # Time step size (derived from data)
    
    function NBodyData(positions, timesteps, times, star_ids)
        # Calculate dt from the time data
        dt = length(times) > 1 ? times[2] - times[1] : 1000.0
        new(positions, sort(timesteps), sort(times), sort(star_ids), dt)
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
- `compatibility_func::Symbol = compositionally_similar_planets`: Name of function to use for deciding what `Planet`s are compatible for future terraformation. 
- `compatibility_kwargs::Union{Dict{Symbol},Nothing} = nothing`: kwargs to pass to `compatibility_func`.
- `destination_func::Symbol = nearest_planet`:  Name of function to use for deciding which compatible `Planet` (which of the `planet.candidate_planet`s) should be the next destination. 
- `pos::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}}`: the initial positions of all `Planet`s.
- `vel::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}}`: the initial velocities of all `Planet`s.
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
    nbody_data
    space_offset

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
        compatibility_func::Symbol = :compositionally_similar,
        compatibility_kwargs::Union{Dict{Symbol},Nothing} = nothing,
        destination_func::Symbol = :nearest,
        pos::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}},
        vel::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}},
        maxcomp::Real,
        compsize::Int,
        planetcompositions::Array{<:Real, 2},
        # New fields for N-body integration (optional)
        nbody_data::Union{NBodyData,Nothing} = nothing,
        ) where {D}

        pos_dims = length(first(pos))
        if !all(p -> length(p) == pos_dims, pos)
            throw(ArgumentError("All positions must have the same dimensionality"))
        end

        vel_dims = length(first(vel))
        if !all(v -> length(v) == vel_dims, vel)
            throw(ArgumentError("All velocities must have the same dimensionality"))
        end

        if pos_dims != vel_dims
            throw(ArgumentError("pos and vel must have the same dims"))
        end

        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        if ~all(x->length(x)==compsize, eachcol(planetcompositions))
            throw(ArgumentError("All planets compositions must have length of `compsize`"))
        end
        
        ## ABMkwargs
        if ABMkwargs === nothing 
            ABMkwargs = Dict(:rng => rng, :warn => false)
        elseif haskey(ABMkwargs, :rng)
            rng != ABMkwargs[:rng] && throw(ArgumentError("rng and ABMkwargs[:rng] do not match. ABMkwargs[:rng] will inherit from rng if ABMkwargs[:rng] not provided."))
        else
            ABMkwargs[:rng] = rng
        end

        ## SpaceArgs
        if SpaceArgs === nothing
            SpaceArgs = Dict(:extent => extent)
        elseif haskey(SpaceArgs, :extent)  # ← Change this line
            extent != SpaceArgs[:extent] && throw(ArgumentError("extent and SpaceArgs[:extent] do not match. SpaceArgs[:extent] will inherit from extent if SpaceArgs[:extent] not provided."))
        else
            SpaceArgs[:extent] = extent
        end

        ## SpaceKwargs
        SpaceKwargs === nothing && (SpaceKwargs = Dict(:periodic => true))

        # Convert positions to SVectors if they're not already
        if !(first(pos) isa SVector)
            pos = [SVector{pos_dims, Float64}(p) for p in pos]
        end
        
        # Convert velocities to SVectors if they're not already
        if !(first(vel) isa SVector)
            vel = [SVector{vel_dims, Float64}(v) for v in vel]
        end

        
        new(rng, extent, ABMkwargs, SpaceArgs, SpaceKwargs, dt, lifespeed, interaction_radius, ool, nool, spawn_rate, compmix_func, compmix_kwargs, compatibility_func, compatibility_kwargs, destination_func, pos, vel, maxcomp, compsize, planetcompositions, nbody_data)

    end
    
end


"""
    GalaxyParameters(rng::AbstractRNG;
        pos::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}, Nothing} = nothing,
        vel::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}, Nothing} = nothing,
        planetcompositions::Union{Array{<:Real,2}, Nothing} = nothing,
        kwargs...) where {D}

Can be called with only `rng` and one of `pos`, `vel` or `planetcompositions`, plus any number of optional kwargs.

# Notes:
Uses GalaxyParameters(rng::AbstractRNG, nplanets::Int; ...) constructor for other arguments
"""
function GalaxyParameters(rng::AbstractRNG;
    pos::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}, Nothing} = nothing,
    vel::Union{Vector{<:NTuple{D,Real}}, Vector{<:AbstractVector{<:Real}}, Nothing} = nothing,
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

"""
    GalaxyParameters(rng::AbstractRNG, nbody_data::NBodyData; kwargs...)

Constructor for N-body simulations where planet count, positions, and simulation extent 
are automatically determined from pre-calculated N-body trajectory data.

# Arguments
- `rng::AbstractRNG`: Random number generator for composition generation and other stochastic processes
- `nbody_data::NBodyData`: Pre-loaded N-body simulation data containing star trajectories

# Keyword Arguments
- `extent::Union{NTuple{3,<:Real}, Nothing} = nothing`: Simulation space extent. If `nothing`, 
  automatically calculated from N-body data bounds with padding
- `spacing::Union{Real, Nothing} = nothing`: Grid spacing for ContinuousSpace. If `nothing`, 
  set to `minimum(extent)/20` and adjusted to ensure extent divisibility
- `space_offset::Union{SVector{3,Float64}, Nothing} = nothing`: Coordinate transformation offset 
  from N-body world coordinates to simulation coordinates. If `nothing`, calculated as minimum 
  coordinates from N-body data
- `maxcomp::Real = 10`: Maximum value for randomly generated planet composition elements
- `compsize::Int = 10`: Length of planet composition vectors
- `padding_factor::Float64 = 0.1`: Fraction of data range to add as padding when calculating 
  automatic extent (e.g., 0.1 = 10% padding on each side)
- `kwargs...`: Additional parameters passed to the main GalaxyParameters constructor

# Returns
`GalaxyParameters` object configured for N-body simulation with:
- Number of planets matching number of stars in N-body data
- Planet IDs corresponding to star IDs in N-body data
- Initial positions from first timestep of N-body data
- Zero velocities (positions come from N-body data, not physics simulation)
- Automatically sized simulation extent with appropriate spacing
- Random planet compositions

# Notes
- This constructor automatically handles coordinate system conversion between N-body simulation 
  coordinates and Agents.jl simulation coordinates
- Planet positions will be updated each timestep from N-body data during simulation
- Requires use of `galaxy_agent_step_nbody!` and `galaxy_model_step_nbody!` step functions
- Simulation length is limited by the number of timesteps in the N-body data

# Example
```julia
nbody_data = load_nbody_data("trajectory.csv")
rng = MersenneTwister(1234)
params = GalaxyParameters(rng, nbody_data; 
                         lifespeed=0.001, 
                         padding_factor=0.15)
model = galaxy_model_setup(params, galaxy_agent_step_nbody!, galaxy_model_step_nbody!)
```

Related functions: `load_nbody_data`, `galaxy_agent_step_nbody!`, `galaxy_model_step_nbody!`
"""
function GalaxyParameters(rng::AbstractRNG, nbody_data::NBodyData;
    extent::Union{NTuple{3,<:Real}, Nothing} = nothing,
    spacing::Union{Real, Nothing} = nothing,
    space_offset::Union{SVector{3,Float64}, Nothing} = nothing,
    maxcomp=10, 
    compsize=10,
    padding_factor=0.1,
    kwargs...)
    
    # Calculate optimal extent, spacing, and offset if not provided
    if extent === nothing
        extent, calculated_spacing, calculated_offset = calculate_optimal_extent(nbody_data; padding_factor, spacing)
        
        if spacing === nothing
            spacing = calculated_spacing
        end
        if space_offset === nothing
            space_offset = calculated_offset
        end
    else
        @info "Using user-provided extent: $extent"
        if spacing === nothing
            spacing = minimum(extent) / 20.0
        end
        if space_offset === nothing
            # Calculate offset logic here...
            initial_timestep = minimum(nbody_data.timesteps)
            first_pos = get_nbody_position(nbody_data, nbody_data.star_ids[1], initial_timestep)
            min_coords = collect(first_pos)
            for star_id in nbody_data.star_ids[2:end]
                pos = get_nbody_position(nbody_data, star_id, initial_timestep)
                for i in 1:3
                    min_coords[i] = min(min_coords[i], pos[i])
                end
            end
            space_offset = SVector{3,Float64}(min_coords...)
        end
    end
    
    # Add spacing to SpaceArgs but don't include space_offset
    args = Dict{Symbol,Any}(kwargs)
    if haskey(args, :SpaceArgs)
        args[:SpaceArgs][:spacing] = spacing
    else
        args[:SpaceArgs] = Dict{Symbol,Union{Real,Tuple}}(:spacing => spacing)
    end
    
    # Rest of constructor...
    nplanets = length(nbody_data.star_ids)
    initial_timestep = minimum(nbody_data.timesteps)
    
    initial_positions = [get_nbody_position(nbody_data, star_id, initial_timestep) 
                        for star_id in nbody_data.star_ids]
    
    pos = [SVector{3, Float64}(p) for p in initial_positions]
    vel = [SVector{3, Float64}(0, 0, 0) for _ in 1:nplanets]
    planetcompositions = random_compositions(rng, maxcomp, compsize, nplanets)
    
    # Create the GalaxyParameters object
    params = GalaxyParameters(; rng=rng, extent=extent, pos, vel, maxcomp, compsize, 
                             planetcompositions, nbody_data, args...)
    
    # Store space_offset in the created object (add this field to GalaxyParameters struct)
    params.space_offset = space_offset
    
    return params
end

# Convenience constructor without explicit RNG
function GalaxyParameters(nbody_data::NBodyData; kwargs...)
    GalaxyParameters(Random.default_rng(), nbody_data; kwargs...)
end

##############################################################################################################################
## CNS5 Catalog Integration
##############################################################################################################################

# Physical constants
const PC_PER_KM_S_PER_YR = 1.0227121650537077e-6  # conversion 1 km/s -> pc/yr

"""
    CNS5Data

Structure to hold processed CNS5 catalog data with positions and velocities.

# Fields
- `positions::Vector{SVector{3,Float64}}`: Cartesian positions in parsecs (ICRS-ish frame)
- `velocities::Vector{SVector{3,Float64}}`: Cartesian velocities in pc/yr
- `star_ids::Vector{Int}`: Original catalog row indices (1-based)
- `catalog_epoch::Float64`: Reference epoch of the catalog (e.g., 2000.0 for J2000)
"""
struct CNS5Data
    positions::Vector{SVector{3,Float64}}
    velocities::Vector{SVector{3,Float64}}
    star_ids::Vector{Int}
    catalog_epoch::Float64
end

# Helper conversion functions
deg2rad_cns5(x) = x * π / 180.0
mas2rad_cns5(x) = x * 1e-3 / 3600.0 * π/180.0

"""
    radec_to_xyz(ra_deg, dec_deg, dist_pc)

Convert RA/Dec (degrees) + distance (pc) to Cartesian coordinates (pc) in ICRS-ish frame.
"""
function radec_to_xyz(ra_deg, dec_deg, dist_pc)
    ra = deg2rad_cns5(ra_deg)
    dec = deg2rad_cns5(dec_deg)
    x = dist_pc * cos(dec) * cos(ra)
    y = dist_pc * cos(dec) * sin(ra)
    z = dist_pc * sin(dec)
    return SVector{3,Float64}(x, y, z)
end

"""
    motions_to_velocity_vec(ra_deg, dec_deg, pmra, pmdec, rv, parallax)

Convert observables to velocity vector in pc/yr.
- pmra, pmdec: proper motions in mas/yr
- rv: radial velocity in km/s
- parallax: parallax in mas
"""
function motions_to_velocity_vec(ra_deg, dec_deg, pmra, pmdec, rv, parallax)
    d_pc = 1000.0 / parallax
    
    # Tangential velocities (km/s): v = 4.74057 * mu_masyr * d_pc
    k = 4.74057e-3
    v_ra_kms  = k * pmra * d_pc
    v_dec_kms = k * pmdec * d_pc
    
    # Unit basis vectors (ICRS-like)
    ra = deg2rad_cns5(ra_deg)
    dec = deg2rad_cns5(dec_deg)
    u_ra  = SVector{3,Float64}(-sin(ra), cos(ra), 0.0)
    u_dec = SVector{3,Float64}(-cos(ra)*sin(dec), -sin(ra)*sin(dec), cos(dec))
    u_rad = SVector{3,Float64}(cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec))
    
    # Combine components to get velocity in km/s, then convert to pc/yr
    v_kms = v_ra_kms * u_ra + v_dec_kms * u_dec + rv * u_rad
    return v_kms * PC_PER_KM_S_PER_YR
end

"""
    parse_vizier_types(csv_path::String)

Parse VizieR column metadata from TSV file to determine proper column types.
Returns a dictionary mapping column names to Julia types.
"""
function parse_vizier_types(csv_path::String)
    types_dict = Dict{String, Type}()
    
    open(csv_path, "r") do file
        for line in eachline(file)
            if startswith(line, "#Column")
                parts = split(line, '\t')
                if length(parts) >= 4
                    col_name = strip(parts[2])
                    format_spec = strip(parts[3])
                    description = strip(parts[4])
                    
                    format_match = match(r"\(([AIFaiflf])(\d+)(?:\.(\d+))?\)", format_spec)
                    if format_match !== nothing
                        format_type = uppercase(format_match.captures[1])[1]
                        nullable = occursin('?', description)
                        
                        if format_type == 'I'
                            base_type = Int
                        elseif format_type == 'F'
                            base_type = Float64
                        else
                            base_type = String
                        end
                        
                        julia_type = nullable ? Union{base_type, Missing} : base_type
                        types_dict[col_name] = julia_type
                    end
                end
            end
        end
    end
    
    return types_dict
end

"""
    load_cns5_catalog(csv_path::String; kwargs...)

Load CNS5 catalog from VizieR TSV file and convert to positions/velocities.

# Arguments
- `csv_path::String`: Path to CNS5 TSV file from VizieR

# Keyword Arguments
- `sample_uncertainty::Bool = false`: If true, sample from observational error distributions
- `rv_default_sigma::Float64 = 30.0`: Default RV uncertainty (km/s) for stars with missing RV
- `max_stars::Union{Int,Nothing} = nothing`: Maximum number of stars to load (for testing)
- `rng::AbstractRNG = Random.default_rng()`: Random number generator for uncertainty sampling
- `catalog_epoch::Float64 = 2000.0`: Reference epoch of the catalog (J2000)

# Returns
`CNS5Data` structure containing positions, velocities, star IDs, and catalog epoch

# Example
```julia
# Load catalog with mean values
cns5_data = load_cns5_catalog("cns5_tab_sep.tsv")

# Load with uncertainty sampling
cns5_data = load_cns5_catalog("cns5_tab_sep.tsv", 
                             sample_uncertainty=true, 
                             rng=MersenneTwister(42))
```
"""
function load_cns5_catalog(csv_path::String;
                          sample_uncertainty::Bool = false,
                          rv_default_sigma::Float64 = 30.0,
                          max_stars::Union{Int,Nothing} = nothing,
                          rng::AbstractRNG = Random.default_rng(),
                          catalog_epoch::Float64 = 2000.0)
    
    println("Loading CNS5 catalog from: $csv_path")
    
    # Parse column types from VizieR metadata
    vizier_types = parse_vizier_types(csv_path)
    println("Parsed $(length(vizier_types)) column types from VizieR metadata")
    
    # Clean whitespace issues in file
    raw_text = read(csv_path, String)
    cleaned_text = replace(raw_text, r" +\t" => "\t")
    cleaned_text = replace(cleaned_text, r"\t +\t" => "\t\t")
    
    # Write to temp file
    temp_path = tempname() * ".tsv"
    write(temp_path, cleaned_text)
    
    # Read with parsed types
    df = CSV.read(temp_path, DataFrame; 
                 delim='\t', 
                 comment="#", 
                 header=1, 
                 skipto=4,
                 missingstring="",
                 stripwhitespace=true,
                 types=vizier_types)
    
    rm(temp_path)
    
    Nrows = size(df, 1)
    println("Loaded $(Nrows) rows from catalog")
    
    # Extract columns directly from VizieR CNS5 data
    ra = df.RAJ2000
    dec = df.DEJ2000
    parallax = df.plx
    parallax_error = df.e_plx
    pmra = df.pmRA
    pmdec = df.pmDE
    pmra_error = df.e_pmRA
    pmdec_error = df.e_pmDE
    rv = df.RV
    rv_error = df.e_RV
    
    # Create validity mask: positive parallax and non-missing coordinates
    valid_mask = .!ismissing.(parallax) .&& (parallax .> 0.0) .&& 
                 .!ismissing.(ra) .&& .!ismissing.(dec)
    
    println("Valid stars: $(sum(valid_mask)) / $(Nrows)")
    if sum(.!valid_mask) > 0
        println("Filtered out $(sum(.!valid_mask)) stars with invalid astrometry")
    end
    
    # Limit to requested number if specified
    idxs = findall(valid_mask)
    if max_stars !== nothing && length(idxs) > max_stars
        idxs = idxs[1:max_stars]
        println("Limited to first $max_stars stars")
    end
    
    M = length(idxs)
    println("Processing $M stars")
    
    # Process each star
    positions = Vector{SVector{3,Float64}}(undef, M)
    velocities = Vector{SVector{3,Float64}}(undef, M)
    star_ids = Vector{Int}(undef, M)
    
    for (i, rowidx) in enumerate(idxs)
        star_ids[i] = rowidx
        
        # Get values for this star
        ra_i = ra[rowidx]
        dec_i = dec[rowidx]
        plx = parallax[rowidx]
        pmr = pmra[rowidx]
        pmd = pmdec[rowidx]
        rv_i = rv[rowidx]
        
        # Handle errors
        e_plx = (ismissing(parallax_error[rowidx]) || parallax_error[rowidx] == 0.0) ? 
                max(0.05, abs(0.01 * plx)) : parallax_error[rowidx]
        e_pmr = (ismissing(pmra_error[rowidx]) || pmra_error[rowidx] isa Number) ? 
                pmra_error[rowidx] : 0.1
        e_pmd = (ismissing(pmdec_error[rowidx]) || pmdec_error[rowidx] isa Number) ? 
                pmdec_error[rowidx] : 0.1
        e_rv = ismissing(rv_error[rowidx]) ? rv_default_sigma : rv_error[rowidx]
        rv_central = ismissing(rv_i) ? 0.0 : rv_i
        
        if sample_uncertainty
            # Sample from error distributions
            plx_dist = truncated(Normal(plx, e_plx), 1e-6, Inf)
            pmr_dist = Normal(pmr, e_pmr)
            pmd_dist = Normal(pmd, e_pmd)
            rv_dist = Normal(rv_central, e_rv)
            
            plx_s = rand(rng, plx_dist)
            pmr_s = rand(rng, pmr_dist)
            pmd_s = rand(rng, pmd_dist)
            rv_s = rand(rng, rv_dist)
            
            d_pc = 1000.0 / plx_s
            positions[i] = radec_to_xyz(ra_i, dec_i, d_pc)
            velocities[i] = motions_to_velocity_vec(ra_i, dec_i, pmr_s, pmd_s, rv_s, plx_s)
        else
            # Use mean values
            d_pc = 1000.0 / plx
            positions[i] = radec_to_xyz(ra_i, dec_i, d_pc)
            velocities[i] = motions_to_velocity_vec(ra_i, dec_i, pmr, pmd, rv_central, plx)
        end
    end
    
    println("Successfully processed $M stars")
    if sample_uncertainty
        println("Used uncertainty sampling (Monte Carlo)")
    else
        println("Used mean values (deterministic)")
    end
    
    return CNS5Data(positions, velocities, star_ids, catalog_epoch)
end

"""
    calculate_optimal_extent_from_positions(positions, velocities, t_offset; padding_factor=0.1, spacing=nothing)

Calculate optimal extent for simulation space based on position data and future motion.

# Arguments
- `positions::Vector{SVector{3,Float64}}`: Initial positions
- `velocities::Vector{SVector{3,Float64}}`: Velocities
- `t_offset::Float64`: Time offset to consider (for checking future/past extent)
- `padding_factor::Float64 = 0.1`: Fraction of range to add as padding
- `spacing::Union{Real,Nothing} = nothing`: Grid spacing (auto-calculated if nothing)

# Returns
- `extent::NTuple{3,Float64}`: Simulation space extent
- `spacing::Float64`: Grid spacing adjusted for divisibility
"""
function calculate_optimal_extent_from_positions(positions::Vector{SVector{3,Float64}}, 
                                                velocities::Vector{SVector{3,Float64}},
                                                t_offset::Float64;
                                                padding_factor::Float64 = 0.1,
                                                spacing::Union{Real,Nothing} = nothing)
    
    # Calculate positions at both t=0 and t=t_offset to capture full range
    positions_at_offset = [p + v * t_offset for (p, v) in zip(positions, velocities)]
    all_positions = vcat(positions, positions_at_offset)
    
    # Find min/max coordinates
    min_coords = [minimum(pos[i] for pos in all_positions) for i in 1:3]
    max_coords = [maximum(pos[i] for pos in all_positions) for i in 1:3]
    
    # Calculate ranges with padding
    ranges = max_coords .- min_coords
    padded_ranges = ranges .* (1 + 2 * padding_factor)
    
    # Calculate spacing if not provided
    if spacing === nothing
        spacing = minimum(padded_ranges) / 20.0
    end
    
    # Calculate number of cells needed and derive exact extent from that
    # This ensures extent = n_cells * spacing exactly
    n_cells = ceil.(Int, padded_ranges ./ spacing)
    adjusted_ranges = n_cells .* spacing
    
    @info "CNS5 extent calculation:" raw_extent=tuple(padded_ranges...) n_cells=tuple(n_cells...) adjusted_extent=tuple(adjusted_ranges...) spacing=spacing
    
    return tuple(adjusted_ranges...), spacing
end

"""
    GalaxyParameters(rng::AbstractRNG, cns5_data::CNS5Data; kwargs...)

Constructor for initializing GalaxyParameters from CNS5 catalog data.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `cns5_data::CNS5Data`: Loaded CNS5 catalog data

# Keyword Arguments
- `t_offset::Float64 = 0.0`: Time offset in years from catalog epoch. Negative values 
  go backward in time (e.g., -50000.0 starts 50k years before J2000)
- `extent::Union{NTuple{3,<:Real}, Nothing} = nothing`: Simulation space extent 
  (auto-calculated if nothing)
- `spacing::Union{Real, Nothing} = nothing`: Grid spacing (auto-calculated if nothing)
- `padding_factor::Float64 = 0.1`: Padding factor for auto-calculated extent
- `maxcomp::Real = 10`: Maximum composition value
- `compsize::Int = 10`: Length of composition vectors
- `kwargs...`: Additional parameters passed to main GalaxyParameters constructor

# Returns
`GalaxyParameters` configured for CNS5 simulation

# Example
```julia
# Start 50,000 years in the past
cns5_data = load_cns5_catalog("cns5.tsv")
params = GalaxyParameters(rng, cns5_data; 
                         t_offset = -50000.0,
                         dt = 100.0,
                         lifespeed = 0.001)
model = galaxy_model_setup(params, galaxy_agent_step_spawn_at_rate!, galaxy_model_step!)

# Run for 500 steps (50k years) to reach present day
run!(model, 500)
```
"""
function GalaxyParameters(rng::AbstractRNG, cns5_data::CNS5Data;
    t_offset::Float64 = 0.0,
    extent::Union{NTuple{3,<:Real}, Nothing} = nothing,
    spacing::Union{Real, Nothing} = nothing,
    padding_factor::Float64 = 0.1,
    maxcomp::Real = 10,
    compsize::Int = 10,
    kwargs...)
    
    @info "Initializing GalaxyParameters from CNS5 data" n_stars=length(cns5_data.star_ids) t_offset=t_offset catalog_epoch=cns5_data.catalog_epoch
    
    # Apply time offset to positions
    # If t_offset = -50000, positions are rewound 50k years into the past
    pos = [SVector{3,Float64}(p + v * t_offset) 
           for (p, v) in zip(cns5_data.positions, cns5_data.velocities)]
    vel = cns5_data.velocities
    
    # Calculate extent if not provided
    if extent === nothing
        extent, calculated_spacing = calculate_optimal_extent_from_positions(
            cns5_data.positions, cns5_data.velocities, t_offset; 
            padding_factor, spacing)

        @show extent
        @show calculated_spacing
        @show extent./calculated_spacing
        
        if spacing === nothing
            spacing = calculated_spacing
        end
    else
        @info "Using user-provided extent: $extent"
        if spacing === nothing
            spacing = minimum(extent) / 20.0
        end
    end
    
    # Set up SpaceArgs with spacing
    args = Dict{Symbol,Any}(kwargs)
    if haskey(args, :SpaceArgs)
        args[:SpaceArgs][:spacing] = spacing
    else
        args[:SpaceArgs] = Dict{Symbol,Union{Real,Tuple}}(:spacing => spacing)
    end
    
    # Generate random compositions
    nplanets = length(cns5_data.star_ids)
    planetcompositions = random_compositions(rng, maxcomp, compsize, nplanets)
    
    @info "CNS5 initialization complete" nplanets=nplanets extent=extent spacing=spacing
    
    # Call main constructor
    return GalaxyParameters(; rng=rng, extent=extent, pos, vel, maxcomp, compsize, 
                           planetcompositions, args...)
end

# Convenience constructor without explicit RNG
function GalaxyParameters(cns5_data::CNS5Data; kwargs...)
    GalaxyParameters(Random.default_rng(), cns5_data; kwargs...)
end

##############################################################################################################################
## NBody Core
##############################################################################################################################
# CORE DATA STRUCTURES AND LOADING
"""
    load_nbody_data(csv_path::String)

Load N-body simulation data from CSV file and return NBodyData structure.
CSV should have columns: star, timestep, time, x, y, z
"""
function load_nbody_data(csv_path::String)
    println("Loading N-body data from: $csv_path")
    
    df = CSV.read(csv_path, DataFrame)
    
    # Validate required columns
    required_cols = [:star, :timestep, :time, :x, :y, :z]
    missing_cols = setdiff(required_cols, propertynames(df))
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    
    # Build position dictionary
    positions = Dict{Tuple{Int,Int}, SVector{3,Float64}}()
    
    println("Processing $(nrow(df)) position records...")
    for row in eachrow(df)
        star_id = Int(row.star)
        timestep = Int(row.timestep)
        pos = SVector{3,Float64}(row.x, row.y, row.z)
        positions[(star_id, timestep)] = pos
    end
    
    # Extract metadata
    timesteps = sort(unique(df.timestep))
    times = sort(unique(df.time))
    star_ids = sort(unique(df.star))
    
    println("Loaded data for $(length(star_ids)) stars over $(length(timesteps)) timesteps")
    println("Time range: $(minimum(times)) to $(maximum(times))")
    
    return NBodyData(positions, timesteps, times, star_ids)
end

"""
    get_nbody_position(nbody_data::NBodyData, star_id::Int, timestep::Int)

Get position of a star at a specific timestep from N-body data.
"""
function get_nbody_position(nbody_data::NBodyData, star_id::Int, timestep::Int)
    key = (star_id, timestep)
    if !haskey(nbody_data.positions, key)
        error("Position not found for star $star_id at timestep $timestep")
    end
    return nbody_data.positions[key]
end

function calculate_reachable_destinations_nbody(planet::Planet, model::ABM, current_timestep::Int)
    basic_candidates = basic_candidate_planets(planet, model)
    
    if !haskey(abmproperties(model), :nbody_data) || model.nbody_data === nothing
        return basic_candidates
    end
    
    nbody_data = model.nbody_data
    
    # Check if we can launch (not at final timestep)
    arrival_timestep = current_timestep + 1
    if arrival_timestep > maximum(nbody_data.timesteps)
        return Planet[]
    end
    
    # Use agent.id directly as star_id
    if !(planet.id in nbody_data.star_ids)
        @warn "Planet $(planet.id) not found in N-body data"
        return Planet[]
    end
    
    origin_pos = get_nbody_position(nbody_data, planet.id, current_timestep)
    max_distance = model.lifespeed * nbody_data.dt  # Use existing lifespeed
    
    reachable_candidates = Planet[]
    for candidate in basic_candidates
        if candidate.id in nbody_data.star_ids
            try
                dest_pos = get_nbody_position(nbody_data, candidate.id, arrival_timestep)
                travel_distance = norm(dest_pos - origin_pos)
                
                if travel_distance <= max_distance
                    push!(reachable_candidates, candidate)
                end
            catch
                continue  # Skip if position not available
            end
        end
    end
    
    return reachable_candidates
end

"""
    spawnlife_mission!(planet::Planet, model::ABM, destination_planet::Planet, parent_life::Union{Life,Nothing})

Create a mission-style life agent for N-body simulations.
This creates a Life agent that will "teleport" to destination after travel time.
"""
function spawnlife_mission!(planet::Planet, model::ABM, destination_planet::Planet, parent_life::Union{Life,Nothing})
    current_timestep = model.current_nbody_timestep
    
    # Create life agent at planet position but as a "mission"
    life = Life(;
        id = Agents.nextid(model),
        pos = planet.pos,
        vel = SVector{3,Float64}(0, 0, 0),  # No velocity needed for missions
        parentplanet = planet,
        composition = planet.composition,
        destination = destination_planet,
        destination_distance = 0.0,  # Not used for missions
        ancestors = get_ancestors(parent_life),
        
        # Mission-specific fields
        departure_timestep = current_timestep,
        arrival_timestep = current_timestep + 1,
        is_mission = true
    )
    
    add_agent_own_pos!(life, model)
    
    # Mark destination as claimed and update launch tracking
    destination_planet.claimed = true
    planet.last_launch_timestep = current_timestep
    
    return model
end

"""
    initialize_nbody_properties!(model::ABM, params::GalaxyParameters)

Initialize N-body specific model properties and update planet positions.
"""
function initialize_nbody_properties!(model::ABM, params::GalaxyParameters)
    # Space offset already calculated and stored in model properties
    # Just update planet positions
    initial_timestep = model.current_nbody_timestep
    
    for planet in filter_agents(model, Planet)
        world_pos = get_nbody_position(model.nbody_data, planet.id, initial_timestep)
        planet.pos = world_pos - model.space_offset
    end
    
    @info "N-body properties initialized for $(length(model.nbody_data.star_ids)) planets"
end

# HELPER FUNCTIONS (these support the main integration functions):
"""
Helper functions to reuse existing compatibility and destination logic
"""
function apply_compatibility_function(planet::Planet, candidates::Vector{Planet}, model::ABM)
    # Reuse your existing compatibility function dispatch logic
    if model.compatibility_func == :compositionally_similar
        return compositionally_similar_planets_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :planets_in_range
        return planets_in_range_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :nearest_k
        return nearest_k_planets_static(planet, candidates; model.compatibility_kwargs...)
    else
        error("Unknown compatibility function: $(model.compatibility_func)")
    end
end

function apply_destination_function(planet::Planet, candidates::Vector{Planet}, model::ABM)
    # Reuse your existing destination function dispatch logic
    if model.destination_func == :nearest
        return nearest_planet_static(planet, candidates)
    elseif model.destination_func == :most_similar
        return most_similar_planet_static(planet, candidates)
    else
        error("Unknown destination function: $(model.destination_func)")
    end
end

"""
    calculate_optimal_extent(nbody_data::NBodyData; padding_factor=0.1, spacing=nothing)

Calculate optimal extent and space offset for simulation space based on N-body data boundaries.
Returns both extent and the coordinate offset needed for translation.
"""
function calculate_optimal_extent(nbody_data::NBodyData; padding_factor=0.1, spacing=nothing)
    # Get all positions across all timesteps
    all_positions = collect(values(nbody_data.positions))
    
    if isempty(all_positions)
        error("No position data found in N-body data")
    end
    
    # Find min/max coordinates across all dimensions in single pass
    min_coords = [minimum(pos[i] for pos in all_positions) for i in 1:3]
    max_coords = [maximum(pos[i] for pos in all_positions) for i in 1:3]
    
    # Calculate ranges for each dimension
    ranges = max_coords .- min_coords
    
    # Add padding (default 10% on each side)
    padded_ranges = ranges .* (1 + 2 * padding_factor)
    
    # If no spacing provided, calculate a reasonable default
    if spacing === nothing
        spacing = minimum(padded_ranges) / 20.0
    end
    
    # Round up each dimension to be exactly divisible by spacing
    adjusted_ranges = ceil.(padded_ranges ./ spacing) .* spacing
    
    # Create space offset from minimum coordinates
    space_offset = SVector{3,Float64}(min_coords...)
    
    @info "Raw extent:\t$(tuple(padded_ranges...))"
    @info "Adjusted extent:\t$(tuple(adjusted_ranges...))"
    @info "Spacing:\t$spacing"
    @info "Space offset:\t$space_offset"
    
    # Return extent, spacing, and offset
    return tuple(adjusted_ranges...), spacing, space_offset
end
# OPTIONAL UTILITY FUNCTIONS (useful for testing and analysis):
# - validate_nbody_data(nbody_data::NBodyData)
# - analyze_reachability(nbody_data::NBodyData, max_velocity::Float64, sample_timesteps::Int=10)
# - create_test_nbody_data(n_stars::Int=10, n_timesteps::Int=100; kwargs...)
# - run_quick_test(csv_path::Union{String,Nothing}=nothing)


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
function center_position(
    pos::Union{<:NTuple{D,Real}, <:AbstractVector{<:Real}}, 
    extent::Union{<:NTuple{D,Real}, <:AbstractVector{<:Real}}, 
    m::Real) where {D} 
    
    pos.+((extent.-(extent./m))./2) 
end


"""

    center_and_scale(pos::NTuple{D,Real}, extent::NTuple{D,Real}, m::Real) where {D}

Assuming that the provided position is for the original `extent` size, 
return the equivilent position in a volume the size of extent/m centered in the extent.

The purpose for this is to allow room for the stars to move.
"""
function center_and_scale(
    pos::Union{<:NTuple{D,Real}, <:AbstractVector{<:Real}}, 
    extent::Union{<:NTuple{D,Real}, <:AbstractVector{<:Real}}, 
    m::Real) where {D} 

    # (pos./m) .+ (extent./2)  .- (m/2)
    pos./m .+ (extent./2) .- (extent./m./2)

end

"""
    galaxy_model_setup(params::GalaxyParameters, agent_step!, model_step!)

Setup function that respects the user's explicit choice of step functions.
For N-body simulations, user must explicitly provide galaxy_agent_step_nbody! and galaxy_model_step_nbody!.
"""
function galaxy_model_setup(params::GalaxyParameters, agent_step!, model_step!)
    println("Setting up galaxy...")
    # Validate that N-body mode uses appropriate step functions
    if params.nbody_data !== nothing
        if agent_step! != galaxy_agent_step_nbody! || model_step! != galaxy_model_step_nbody!
            error("""
            N-body mode detected but incompatible step functions provided.
            For N-body simulations, you must use:
              agent_step! = galaxy_agent_step_nbody!
              model_step! = galaxy_model_step_nbody!
            
            This is required because N-body simulations use pre-calculated positions
            rather than physics-based movement.
            """)
        end
        @info "N-body mode: Using pre-calculated planet trajectories"
    end
    
    println("Setting up planets...")
    model = galaxy_planet_setup(params, agent_step!, model_step!)
    
    # Add N-body properties if data provided
    if params.nbody_data !== nothing
        println("Initializing N-body properties...")
        initialize_nbody_properties!(model, params)
    end
    
    println("Setting up life...")
    model = galaxy_life_setup(model, params)
    println("Model setup complete.")
    model
end
"""
    galaxy_model_setup(params::Dict)

Initializes the GalaxyParameters struct from the provided dict.
"""
function galaxy_model_setup(params::Dict, agent_step!, model_step!)

    params = GalaxyParameters(
        params[:rng],
        params[:nplanets]; 
        filter(x -> first(x) ∉ [:rng, :nplanets], params)...)
    model = galaxy_planet_setup(params, agent_step!, model_step!)
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
function galaxy_planet_setup(params::GalaxyParameters, agent_step!, model_step!)

    extent_multiplier = 1
    params.extent = extent_multiplier.*params.extent

    if haskey(params.SpaceArgs, :spacing)
        space = ContinuousSpace(params.extent; 
                        spacing=params.SpaceArgs[:spacing], 
                        params.SpaceKwargs...)
    else
        space = ContinuousSpace(params.extent; params.SpaceKwargs...)
    end

    properties = Dict(:dt => params.dt,
                :lifespeed => params.lifespeed,
                :interaction_radius => params.interaction_radius,
                :nplanets => nplanets(params),
                :maxcomp => params.maxcomp,
                :compsize => params.compsize,
                :s => 0,
                :n_living_planets => params.nool,
                :terraformed_on_step => true,
                :n_terraformed_on_step => params.nool,
                :spawn_rate => params.spawn_rate,
                :GalaxyParameters => params,
                :compmix_func => params.compmix_func,
                :compmix_kwargs => params.compmix_kwargs,
                :compatibility_func => params.compatibility_func,
                :compatibility_kwargs => params.compatibility_kwargs,
                :destination_func => params.destination_func)

    # Add N-body properties if N-body data is provided
    if params.nbody_data !== nothing
        properties[:nbody_data] = params.nbody_data
        properties[:current_nbody_timestep] = minimum(params.nbody_data.timesteps)
        properties[:space_offset] = params.space_offset  # Use pre-calculated value
    end

    model = @suppress_err StandardABM(
        Union{Planet,Life},
        space;
        scheduler = allocated_fastest,
        properties = properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
        params.ABMkwargs...
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

        spawn_if_candidate_planets!(planet, model)

    end

    model

end

"""
    initialize_planets!(model, params::GalaxyParameters, extent_multiplier)

Initialize `Planet`s in the galaxy.

`Planet` positions are adjusted to `center_position`, based on `extent_multiplier`.

This acts to increase the space seen by the user when plotting, and put the simulation in the center of the space.

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

function planet_attribute_as_matrix(planets::Vector{<:Planet}, attr::Symbol)
    length(planets) == 0 && throw(ArgumentError("planets is empty"))
    planet_attributes = map(x -> getproperty(x, attr), planets)
    ## need to collect because when attr = :pos, the result is a Vector of Tuples
    convert(Matrix{Float64}, hcat(collect.(planet_attributes)...))
end

##########################################################################################
# Static Planet Functions (original logic)
##########################################################################################

function compositionally_similar_planets_static(planet::Planet, candidates::Vector{<:Planet}; allowed_diff)
    length(candidates) == 0 && return Planet[]
    planets_in_attribute_range(planet, candidates, :composition, allowed_diff)
end

function nearest_k_planets_static(planet::Planet, candidates::Vector{<:Planet}; k)
    n_candidates = length(candidates)
    if n_candidates == 0
        return Planet[]
    elseif k > n_candidates
        k = n_candidates
    end
    
    planetpositions = planet_attribute_as_matrix(candidates, :pos)
    idxs, dists = knn(KDTree(planetpositions), Vector(planet.pos), k)
    candidates[idxs]
end

function planets_in_range_static(planet::Planet, candidates::Vector{<:Planet}; r)
    length(candidates) == 0 && return Planet[]
    planets_in_attribute_range(planet, candidates, :pos, r)
end

function planets_in_attribute_range(planet::Planet, planets::Vector{<:Planet}, attr::Symbol, r)
    planetattributes = planet_attribute_as_matrix(planets, attr)
    idxs = inrange(KDTree(planetattributes), Vector(getproperty(planet, attr)), r)
    planets[idxs]
end

# Static destination functions
function nearest_planet_static(planet::Planet, candidates::Vector{<:Planet})
    closest_planet_by_attribute(planet, candidates, :pos)
end

function most_similar_planet_static(planet::Planet, candidates::Vector{<:Planet})
    closest_planet_by_attribute(planet, candidates, :composition)
end

function closest_planet_by_attribute(planet::Planet, planets::Vector{<:Planet}, attr::Symbol)
    planetattributes = planet_attribute_as_matrix(planets, attr)
    idx, dist = nn(KDTree(planetattributes), Vector(getproperty(planet, attr)))
    planets[idx]
end

##########################################################################################
# Moving Planet Functions (with interception calculations)
##########################################################################################

function compositionally_similar_planets_moving(planet::Planet, candidates::Vector{<:Planet}, model::ABM; allowed_diff)
    length(candidates) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    # Calculate all interceptions once
    velocities_dict, times_dict = calculate_interceptions_exhaustive(planet.pos, model.lifespeed, candidates)
    
    # Only consider planets that have valid interceptions
    valid_planet_ids = collect(keys(times_dict))
    length(valid_planet_ids) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    # Filter candidates to only those with valid interceptions
    reachable_candidates = [p for p in candidates if p.id in valid_planet_ids]
    
    # Filter by composition among the reachable candidates
    compatible_planets = planets_in_attribute_range(planet, reachable_candidates, :composition, allowed_diff)
    
    # Get velocities and times for the compatible planets
    compatible_velocities = [velocities_dict[p.id] for p in compatible_planets]
    compatible_times = [times_dict[p.id] for p in compatible_planets]
    
    return (compatible_planets, compatible_velocities, compatible_times)
end

function planets_within_travel_time_moving(planet::Planet, candidates::Vector{<:Planet}, model::ABM; max_time)
    length(candidates) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    velocities_dict, times_dict = calculate_interceptions_exhaustive(planet.pos, model.lifespeed, candidates)
    
    # Filter by travel time
    valid_entries = [(id, time) for (id, time) in times_dict if time < max_time]
    length(valid_entries) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    valid_ids = [entry[1] for entry in valid_entries]
    valid_planets = [p for p in candidates if p.id in valid_ids]
    valid_velocities = [velocities_dict[id] for id in valid_ids]
    valid_times = [times_dict[id] for id in valid_ids]
    
    return (valid_planets, valid_velocities, valid_times)
end

function planets_in_range_moving(planet::Planet, candidates::Vector{<:Planet}, model::ABM; r)
    length(candidates) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    velocities_dict, times_dict = calculate_interceptions_exhaustive(planet.pos, model.lifespeed, candidates)
    
    # Only consider planets that have valid interceptions
    valid_planet_ids = collect(keys(times_dict))
    length(valid_planet_ids) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    # Filter candidates to only those with valid interceptions
    reachable_candidates = [p for p in candidates if p.id in valid_planet_ids]
    
    # Filter by position range among the reachable candidates
    compatible_planets = planets_in_range_static(planet, reachable_candidates, r)
    
    # Get velocities and times for the compatible planets
    compatible_velocities = [velocities_dict[p.id] for p in compatible_planets]
    compatible_times = [times_dict[p.id] for p in compatible_planets]
    
    return (compatible_planets, compatible_velocities, compatible_times)
end

function nearest_k_planets_by_travel_time_moving(planet::Planet, candidates::Vector{Planet}, model::ABM; k)
    length(candidates) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    velocities_dict, times_dict = calculate_interceptions_exhaustive(planet.pos, model.lifespeed, candidates)
    
    # Build arrays in the same order - use candidates order as the reference
    valid_data = []
    for candidate in candidates
        if candidate.id in keys(times_dict)
            push!(valid_data, (candidate, velocities_dict[candidate.id], times_dict[candidate.id]))
        end
    end
    
    length(valid_data) == 0 && return (Planet[], Vector{Vector{Float64}}[], Float64[])
    
    k = min(k, length(valid_data))
    
    # Sort by travel time (third element of tuple)
    sorted_data = partialsort(valid_data, 1:k, by=x->x[3])
    
    # Extract sorted arrays
    result_planets = [item[1] for item in sorted_data]
    result_velocities = [item[2] for item in sorted_data]
    result_times = [item[3] for item in sorted_data]
    
    return (result_planets, result_velocities, result_times)
end

# Moving destination functions
function nearest_planet_moving(planets::Vector{<:Planet}, velocities::Vector{Vector{Float64}}, times::Vector{Float64})
    length(planets) == 0 && return (nothing, nothing)
    
    min_idx = argmin([norm(p.pos) for p in planets])  # or use times if you prefer
    return (planets[min_idx], velocities[min_idx])
end

function most_similar_planet_moving(reference_planet::Planet, planets::Vector{<:Planet}, velocities::Vector{Vector{Float64}}, times::Vector{Float64})
    length(planets) == 0 && return (nothing, nothing)
    
    # Find most compositionally similar
    composition_diffs = [norm(p.composition .- reference_planet.composition) for p in planets]
    min_idx = argmin(composition_diffs)
    return (planets[min_idx], velocities[min_idx])
end

function fastest_planet_to_reach_moving(planets::Vector{<:Planet}, velocities::Vector{Vector{Float64}}, times::Vector{Float64})
    length(planets) == 0 && return (nothing, nothing)
    
    min_idx = argmin(times)
    return (planets[min_idx], velocities[min_idx])
end


##########################################################################################
# Calculation Strategy Detection
##########################################################################################

function planets_are_static(model::ABM)
    # Cache this in model to avoid recalculating
    if !hasfield(typeof(model), :planets_static_cached)
        return all(p -> all(iszero, p.vel), filter_agents(model, Planet))
    end
    return model.planets_static_cached
end

function update_movement_cache!(model::ABM)
    model.planets_static_cached = all(p -> all(iszero, p.vel), filter_agents(model, Planet))
end

function needs_exhaustive_calculation(compatibility_func::Symbol, destination_func::Symbol)
    # Cases that require knowing ALL travel times upfront
    exhaustive_compatibility = [:within_travel_time, :nearest_k_by_time]
    exhaustive_destination = [:fastest]
    
    return compatibility_func in exhaustive_compatibility || destination_func in exhaustive_destination
end

##########################################################################################
# Main Spawning Functions
##########################################################################################

function spawn_if_candidate_planets!(
    planet::Planet,
    model::ABM,
    life::Union{Life,Nothing} = nothing
)
    if haskey(abmproperties(model), :nbody_data) && model.nbody_data !== nothing
        spawn_if_candidate_planets_nbody!(planet, model, life)
    elseif planets_are_static(model)
        spawn_static!(planet, model, life)
    else
        spawn_moving!(planet, model, life)
    end
    return model
end

"""
    spawn_if_candidate_planets_nbody!(planet::Planet, model::ABM, life::Union{Life,Nothing} = nothing)

Modified version of existing spawn_if_candidate_planets! that works with N-body data.
This replaces the existing function when N-body mode is active.
"""
function spawn_if_candidate_planets_nbody!(planet::Planet, model::ABM, life::Union{Life,Nothing} = nothing)
    # Check launch rate limiting for N-body mode
    if haskey(abmproperties(model), :nbody_data) && model.nbody_data !== nothing
        current_timestep = model.current_nbody_timestep
        
        # Respect launch rate limits (one launch per timestep)
        if (current_timestep - planet.last_launch_timestep) < 1
            return model
        end
        
        # Get candidates using N-body constraints
        candidates = calculate_reachable_destinations_nbody(planet, model, current_timestep)
    else
        # Fall back to existing behavior
        candidates = basic_candidate_planets(planet, model)
    end
    
    if length(candidates) == 0
        return model
    end
    
    # Use existing compatibility and destination functions with candidates
    compatible_planets = apply_compatibility_function(planet, candidates, model)
    if length(compatible_planets) == 0
        return model
    end
    
    destination_planet = apply_destination_function(planet, compatible_planets, model)
    if destination_planet === nothing
        return model
    end
    
    # Create mission-style life or regular life depending on mode
    if haskey(abmproperties(model), :nbody_data) && model.nbody_data !== nothing
        # N-body mission mode
        spawnlife_mission!(planet, model, destination_planet, life)
    else
        # Regular continuous travel mode
        vel = direction(planet, destination_planet) .* model.lifespeed
        spawnlife!(planet, model, vel, destination_planet; ancestors = get_ancestors(life))
    end
    
    return model
end

function spawn_static!(planet::Planet, model::ABM, life::Union{Life,Nothing})
    candidates = basic_candidate_planets(planet, model)
    length(candidates) == 0 && return model
    
    # Apply compatibility function
    compatible_planets = apply_compatibility_static(planet, candidates, model)
    length(compatible_planets) == 0 && return model
    
    # Select destination
    destination_planet = apply_destination_static(planet, compatible_planets, model)
    
    # Calculate simple velocity
    vel = direction(planet, destination_planet) .* model.lifespeed
    
    spawnlife!(planet, model, vel, destination_planet; ancestors = get_ancestors(life))
    return model
end

function spawn_moving!(planet::Planet, model::ABM, life::Union{Life,Nothing})
    candidates = basic_candidate_planets(planet, model)
    length(candidates) == 0 && return model
    
    if needs_exhaustive_calculation(model.compatibility_func, model.destination_func)
        spawn_moving_exhaustive!(planet, candidates, model, life)
    else
        spawn_moving_lazy!(planet, candidates, model, life)
    end
    return model
end

function spawn_moving_exhaustive!(planet::Planet, candidates::Vector{<:Planet}, model::ABM, life::Union{Life,Nothing})
    # Calculate ALL interceptions upfront
    compatible_planets, velocities, times = apply_compatibility_moving(planet, candidates, model)
    length(compatible_planets) == 0 && return model
    
    # Select destination (returns planet and velocity)
    destination_planet, vel = apply_destination_moving(planet, compatible_planets, velocities, times, model)
    
    if !isnothing(destination_planet) && !isnothing(vel)
        spawnlife!(planet, model, vel, destination_planet; ancestors = get_ancestors(life))
    end
end

function spawn_moving_lazy!(planet::Planet, candidates::Vector{<:Planet}, model::ABM, life::Union{Life,Nothing})
    # Filter candidates without calculating interceptions
    filtered_candidates = apply_compatibility_lazy(planet, candidates, model)
    length(filtered_candidates) == 0 && return model
    
    # Try destinations one by one until we find a reachable one
    destination_candidates = get_destination_candidates_lazy(planet, filtered_candidates, model)
    
    for candidate in destination_candidates
        vel, time = calculate_interception(planet, candidate, model)
        if !isnothing(vel)  # Found a reachable planet
            spawnlife!(planet, model, vel, candidate; ancestors = get_ancestors(life))
            return
        end
    end
    # If we get here, no reachable planets were found
end

##########################################################################################
# Lazy Calculation Functions (for moving planets when exhaustive calc not needed)
##########################################################################################

function apply_compatibility_lazy(planet::Planet, candidates::Vector{<:Planet}, model::ABM)
    # These are the same as static versions since they don't need velocity/time data
    if model.compatibility_func == :compositionally_similar
        isnothing(model.compatibility_kwargs) && error("compositionally_similar requires 'allowed_diff' parameter in compatibility_kwargs")
        return compositionally_similar_planets_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :planets_in_range
        isnothing(model.compatibility_kwargs) && error("planets_in_range requires 'r' parameter in compatibility_kwargs")
        return planets_in_range_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :nearest_k
        isnothing(model.compatibility_kwargs) && error("nearest_k requires 'k' parameter in compatibility_kwargs")
        return nearest_k_planets_static(planet, candidates; model.compatibility_kwargs...)
    else
        error("Compatibility function $(model.compatibility_func) requires exhaustive calculation")
    end
end

function get_destination_candidates_lazy(planet::Planet, candidates::Vector{<:Planet}, model::ABM)
    # Return candidates in the order we should try them
    if model.destination_func == :nearest
        # Sort by distance, try closest first
        distances = [norm(c.pos .- planet.pos) for c in candidates]
        sorted_indices = sortperm(distances)
        return candidates[sorted_indices]
    elseif model.destination_func == :most_similar
        # Sort by composition similarity, try most similar first
        composition_diffs = [norm(c.composition .- planet.composition) for c in candidates]
        sorted_indices = sortperm(composition_diffs)
        return candidates[sorted_indices]
    else
        error("Destination function $(model.destination_func) requires exhaustive calculation")
    end
end

##########################################################################################
# Compatibility and Destination Function Dispatchers
##########################################################################################

function apply_compatibility_static(planet::Planet, candidates::Vector{<:Planet}, model::ABM)
    if model.compatibility_func == :compositionally_similar
        isnothing(model.compatibility_kwargs) && error("compositionally_similar requires 'allowed_diff' parameter in compatibility_kwargs")
        return compositionally_similar_planets_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :planets_in_range
        isnothing(model.compatibility_kwargs) && error("planets_in_range requires 'r' parameter in compatibility_kwargs")
        return planets_in_range_static(planet, candidates; model.compatibility_kwargs...)
    elseif model.compatibility_func == :nearest_k
        isnothing(model.compatibility_kwargs) && error("nearest_k requires 'k' parameter in compatibility_kwargs")
        return nearest_k_planets_static(planet, candidates; model.compatibility_kwargs...)
    else
        error("Unknown compatibility function: $(model.compatibility_func)")
    end
end

function apply_compatibility_moving(planet::Planet, candidates::Vector{<:Planet}, model::ABM)
    if model.compatibility_func == :compositionally_similar
        isnothing(model.compatibility_kwargs) && error("compositionally_similar requires 'allowed_diff' parameter in compatibility_kwargs")
        return compositionally_similar_planets_moving(planet, candidates, model; model.compatibility_kwargs...)
    elseif model.compatibility_func == :planets_in_range
        isnothing(model.compatibility_kwargs) && error("planets_in_range requires 'r' parameter in compatibility_kwargs")
        return planets_in_range_moving(planet, candidates, model; model.compatibility_kwargs...)
    elseif model.compatibility_func == :within_travel_time
        isnothing(model.compatibility_kwargs) && error("within_travel_time requires 'max_time' parameter in compatibility_kwargs")
        return planets_within_travel_time_moving(planet, candidates, model; model.compatibility_kwargs...)
    elseif model.compatibility_func == :nearest_k_by_time
        isnothing(model.compatibility_kwargs) && error("nearest_k_by_time requires 'k' parameter in compatibility_kwargs")
        return nearest_k_planets_by_travel_time_moving(planet, candidates, model; model.compatibility_kwargs...)
    else
        error("Unknown compatibility function: $(model.compatibility_func)")
    end
end

function apply_destination_static(planet::Planet, candidates::Vector{<:Planet}, model::ABM)
    if model.destination_func == :nearest
        return nearest_planet_static(planet, candidates)
    elseif model.destination_func == :most_similar
        return most_similar_planet_static(planet, candidates)
    else
        error("Unknown destination function: $(model.destination_func)")
    end
end

function apply_destination_moving(planet::Planet, candidates::Vector{<:Planet}, velocities::Vector{Vector{Float64}}, times::Vector{Float64}, model::ABM)
    if model.destination_func == :nearest
        return nearest_planet_moving(candidates, velocities, times)
    elseif model.destination_func == :most_similar
        return most_similar_planet_moving(planet, candidates, velocities, times)
    elseif model.destination_func == :fastest
        return fastest_planet_to_reach_moving(candidates, velocities, times)
    else
        error("Unknown destination function: $(model.destination_func)")
    end
end

"""
    spawnlife!(planet::Planet, model::ABM; ancestors::Vector{Life} = Life[])

Spawns `Life` at `planet`.

Called by [`galaxy_model_setup`](@ref) and [`terraform!`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM,
    vel,
    destination_planet;
    ancestors::Vector{Life} = Life[]
    )

    life = Life(;
        id = Agents.nextid(model),
        pos = planet.pos,
        vel = SA[vel...],
        parentplanet = planet,
        composition = planet.composition,
        destination = destination_planet,
        destination_distance = distance(destination_planet.pos, planet.pos),
        ancestors
    ) ## Only "first" life won't have ancestors

    life = add_agent_own_pos!(life, model)

    destination_planet.claimed = true 
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

"""
    check_agent_at_boundary!(agent, model)

Check if an agent has reached the space boundary and update its `reached_boundary` status.
Handles 1D, 2D, and 3D GridSpace automatically.

# Arguments
- `agent`: The agent to check (must have `reached_boundary` field)
- `model`: The ABM model containing the space
- `tolerance`: Small value to account for floating point precision (default: 1e-10)
"""
function check_agent_at_boundary!(agent, model; tolerance=1e-10)
    # Skip if already marked as at boundary
    if agent.reached_boundary
        return
    end
    
    space_dims = spacesize(model) #size(model.space)
    pos = agent.pos
    # @show agent.id, pos
    n_dims = length(space_dims)
    
    # Check boundary for each dimension
    for dim in 1:n_dims
        if pos[dim] <= tolerance || pos[dim] >= (space_dims[dim] - tolerance)
            agent.reached_boundary = true
            return
        end
    end
end
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
function galaxy_model_step_core!(model)
    
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
    galaxy_model_step!(model::ABM)

Modified model step function that advances N-body timestep when in N-body mode.
This can replace your existing model step function.
"""
function galaxy_model_step!(model::ABM)
    # Handle N-body timestep advancement
    if haskey(abmproperties(model), :nbody_data) && model.nbody_data !== nothing
        # Advance N-body timestep
        current_timestep = model.current_nbody_timestep
        model.current_nbody_timestep = current_timestep + 1
        
        # Set model dt to match N-body data
        model.dt = model.nbody_data.dt
        
        # Check if we've reached end of N-body data
        if model.current_nbody_timestep > maximum(model.nbody_data.timesteps)
            println("Reached end of N-body data at timestep $(model.current_nbody_timestep)")
        end
    end
    
    # Rest of existing model step logic
    galaxy_model_step_core!(model)
end

"""
    galaxy_model_step_nbody!(model::ABM)

N-body model step function that advances N-body timestep.
"""
function galaxy_model_step_nbody!(model::ABM)
    # Handle N-body timestep advancement
    if haskey(abmproperties(model), :nbody_data) && model.nbody_data !== nothing
        current_timestep = model.current_nbody_timestep
        model.current_nbody_timestep = current_timestep + 1
        
        # Set model dt to match N-body data
        model.dt = model.nbody_data.dt
        
        # Check if we've reached end of N-body data
        if model.current_nbody_timestep > maximum(model.nbody_data.timesteps)
            @info "Reached end of N-body data at timestep $(model.current_nbody_timestep)"
        end
    else
        error("N-body model step called without N-body data")
    end
    
    # Rest of existing model step logic
    galaxy_model_step_core!(model)
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

    ## allow planets to move
    move_agent!(planet, model, model.dt)

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

    move_agent!(planet, model, model.dt)
    check_agent_at_boundary!(planet, model)

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

    move_agent!(planet, model, model.dt)

end

"""
    galaxy_agent_step_nbody!(agent::Union{Planet,Life}, model::ABM)

Modified agent step function that handles N-body mode.
"""
function galaxy_agent_step_nbody!(agent::Planet, model::ABM)
    if !haskey(abmproperties(model), :nbody_data) || model.nbody_data === nothing
        error("N-body agent step called without N-body data")
    end
    
    # Use agent.id directly as star_id
    if agent.id in model.nbody_data.star_ids
        current_timestep = model.current_nbody_timestep
        world_pos = get_nbody_position(model.nbody_data, agent.id, current_timestep)
        agent.pos = world_pos - model.space_offset
    else
        @warn "Planet $(agent.id) not found in N-body data - position not updated"
    end
end

function galaxy_agent_step_nbody!(agent::Life, model::ABM)
    # In N-body mode, all Life agents should be missions
    if agent.is_mission
        # Mission mode: check if arrival time reached
        current_timestep = model.current_nbody_timestep
        if current_timestep >= agent.arrival_timestep
            # Mission arrives - terraform and remove
            terraform!(agent, agent.destination, model)
            spawn_if_candidate_planets_nbody!(agent.destination, model, agent)
            remove_agent!(agent, model)
        end
        # No movement needed for missions - they just wait until arrival time
    else
        # This shouldn't happen in N-body mode
        throw(ArgumentError("Life agent in N-body mode should be a mission but is_mission=false for agent $(agent.id)"))
    end
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
    for planet in filter_agents(model,Planet)

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
    # Create a new dataframe with extracted position components
    pos_columns = DataFrame()
    
    # Define dimension names programmatically
    dim_names = ["x", "y", "z"][1:ndims]
    
    # Extract each dimension from the SVector with proper naming
    for (i, name) in enumerate(dim_names)
        pos_columns[!, Symbol(name)] = [p[i] for p in df_planets.pos]
        vel_columns[!, Symbol("v_"*name)] = [p[i] for p in df_planets.vel]
    end
    
    # Join the position columns to the original dataframe
    df = hcat(df_planets, pos_columns)
    
    # Do the same for composition (no change needed as it's already a Vector)
    df = transform(df, :composition => AsTable)
    new_names = Dict("x$i" => "comp_$(i)" for i in 1:compsize)
    rename!(df, new_names)
    
    # Remove original vector columns
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
## Moving planet specific functions
##############################################################################################################################

function random_stellar_velocities(rng::AbstractRNG=Random.GLOBAL_RNG, 
                                  n::Int=1;
                                  σ_U::Float64=35.0, 
                                  σ_V::Float64=25.0, 
                                  σ_W::Float64=20.0, 
                                  V_lag::Float64=15.0,
                                  scale::Float64=1.0)::Vector{SVector{3, Float64}}
    """
    Generate random stellar velocities following realistic velocity dispersion.
    
    Parameters:
    -----------
    rng : AbstractRNG
        Random number generator to use
    n : Int
        Number of velocity vectors to generate
    σ_U : Float64
        Velocity dispersion in U direction (radial, toward/away from galactic center) in km/s
    σ_V : Float64
        Velocity dispersion in V direction (azimuthal, along galactic rotation) in km/s
    σ_W : Float64
        Velocity dispersion in W direction (vertical, perpendicular to galactic plane) in km/s
    V_lag : Float64
        Asymmetric drift - typical lag behind circular velocity in km/s
    scale : Float64
        Scale factor to multiply all velocity components (useful for unit conversion
        or simulating different stellar populations)
        
    Returns:
    --------
    Vector{SVector{3, Float64}} containing n velocity vectors
    """
    
    # Apply scale factor to dispersions and drift
    σ_U_scaled = σ_U * scale
    σ_V_scaled = σ_V * scale
    σ_W_scaled = σ_W * scale
    V_lag_scaled = V_lag * scale
    
    # Create distributions for each component
    dist_U = Normal(0.0, σ_U_scaled)
    dist_V = Normal(-V_lag_scaled, σ_V_scaled)  # Note negative mean for asymmetric drift
    dist_W = Normal(0.0, σ_W_scaled)
    
    # Generate random velocities and package as StaticVectors
    velocities = Vector{SVector{3, Float64}}(undef, n)
    
    for i in 1:n
        U = rand(rng, dist_U)
        V = rand(rng, dist_V)
        W = rand(rng, dist_W)
        velocities[i] = SVector{3, Float64}(U, V, W)
    end
    
    return velocities
end

function calculate_interception(life::Life)
    calculate_interception(life.pos, speed(life), life.destination.pos, life.destination.vel)
end

function calculate_interception(starting_planet::Planet, destination_planet::Planet, model)
    calculate_interception(starting_planet.pos, model.lifespeed, destination_planet.pos, destination_planet.vel)
end

"""
    calculate_interception(r0, v_agent_speed, r1, v1) -> (v_agent, t)

Calculate the velocity vector and time needed for an agent traveling at constant speed 
to intercept a moving target.

# Arguments
- `r0::AbstractVector`: Agent's initial position vector (works in any dimension)
- `v_agent_speed::Real`: Agent's constant speed (scalar, must be positive)
- `r1::AbstractVector`: Target's initial position vector (same dimension as `r0`)
- `v1::AbstractVector`: Target's velocity vector (same dimension as `r0`)

# Returns
- `v_agent::Union{AbstractVector, Nothing}`: Velocity vector the agent should travel at to intercept the target, 
  or `nothing` if interception is impossible
- `t::Union{Real, Nothing}`: Time until interception, or `nothing` if interception is impossible

# Details
Solves the interception problem by finding the time when the agent and target will be at the 
same position, given that the agent travels at a constant speed. The solution involves solving 
a quadratic equation derived from the constraint that the agent's velocity magnitude is fixed.

For interception to be possible, the discriminant of the quadratic equation must be non-negative.
If multiple intercept times exist, the earliest positive time is chosen.

# Notes
- A warning is issued if the computed velocity magnitude differs from the specified agent speed 
  beyond numerical precision (> 1e-10)
- The function works with vectors of any dimension (2D, 3D, etc.)
- Requires the LinearAlgebra module for dot product and norm calculations
"""
function calculate_interception(r0, v_agent_speed, r1, v1)
    # r0: Agent's initial position (vector)
    # v_agent_speed: Agent's constant speed (scalar)
    # r1: Planet's initial position (vector)
    # v1: Planet's velocity (vector)
    
    # Calculate relative initial position
    dr = r1 - r0
    
    # Set up quadratic equation coefficients: at² + bt + c = 0
    a = dot(v1, v1) - v_agent_speed^2
    b = 2 * dot(dr, v1)
    c = dot(dr, dr)
    
    # Calculate discriminant
    discriminant = b^2 - 4 * a * c
    
    if discriminant < 0
        # No real solution exists - interception is impossible
        return nothing, nothing
    end
    
    # Calculate possible interception times
    t1 = (-b + sqrt(discriminant)) / (2 * a)
    t2 = (-b - sqrt(discriminant)) / (2 * a)
    
    # Choose the smallest positive time
    if t1 > 0 && (t2 <= 0 || t1 < t2)
        t = t1
    elseif t2 > 0
        t = t2
    else
        # No positive time solution - interception is impossible
        return nothing, nothing
    end
    
    # Calculate interception point
    interception_point = r1 + v1 * t
    
    # Calculate required velocity vector for agent
    v_agent = (interception_point - r0) / t
    
    # Verify the speed is correct (within numerical precision)
    speed_error = abs(norm(v_agent) - v_agent_speed)
    if speed_error > 1e-10
        @warn "Speed error: $(speed_error). Solution may be inaccurate."
    end
    
    return v_agent, t
end

function calculate_interceptions_exhaustive(life::Life)
    calculate_interceptions_exhaustive(life.pos, speed(life), life.parentplanet.candidate_planets)
end

"""
    calculate_interceptions_exhaustive(r0, v_agent_speed, planet_agents) -> (vs_agent, ts)

Calculate interception vectors and times for multiple planets using the existing calculate_interception function.

# Arguments
- `r0::AbstractVector`: Interceptor's initial position vector
- `v_agent_speed::Real`: Interceptor's constant speed (scalar, must be positive)
- `planet_agents`: Collection of agents with .pos, .vel, and .id attributes

# Returns
- `vs_agent::Dict{Int,Vector{Float64}}`: Dictionary mapping planet IDs to required velocity vectors
- `ts::Dict{Int,Float64}`: Dictionary mapping planet IDs to interception times

# Details
Simply calls calculate_interception for each planet and collects successful results.
"""
function calculate_interceptions_exhaustive(r0, v_agent_speed, planet_agents)
    vs_agent = Dict{Int, Vector{Float64}}()
    ts = Dict{Int, Float64}()
    
    for planet in planet_agents
        v_agent, t = calculate_interception(r0, v_agent_speed, planet.pos, planet.vel)
        
        if v_agent !== nothing && t !== nothing
            vs_agent[planet.id] = v_agent
            ts[planet.id] = t
        end
    end
    
    return vs_agent, ts
end



##############################################################################################################################
## Interactive Plot utilities 
##############################################################################################################################

"""
Overload Agents.jl's agent2string function with custom fields for Planets

For more information see: https://juliadynamics.github.io/Agents.jl/stable/examples/agents_visualizations/#Agent-inspection
https://stackoverflow.com/questions/37031133/how-do-you-format-a-string-when-interpolated-in-julia
"""
function Agents.agent2string(agent::Planet)
    """
    Planet
    id = $(agent.id)
    pos = ($(join([@sprintf("%.2f", i) for i in agent.pos],", ")))
    vel = ($(join([@sprintf("%.2f", i) for i in agent.vel],", ")))
    composition = [$(join([@sprintf("%.2f", i) for i in agent.composition],", "))]
    initialcomposition = [$(join([@sprintf("%.2f", i) for i in agent.initialcomposition],", "))]
    alive = $(agent.alive)
    claimed = $(agent.claimed)
    parentplanets (†‡): $(length(agent.parentplanets) == 0 ? "No parentplanet" : agent.parentplanets[end].id)
    parentlifes (†‡): $(length(agent.parentlifes) == 0 ? "No parentlife" : agent.parentlifes[end].id)
    parentcompositions (‡): $(length(agent.parentcompositions) == 0 ? "No parentcomposition" : "[$(join([@sprintf("%.2f", i) for i in agent.parentcompositions[end]],", "))]")
    last_launch_timestep: $(@sprintf("%.2f", agent.last_launch_timestep))
    reached boundary: $(agent.reached_boundary)
    """
    ## Have to exclude this because it's taking up making the rest of the screen invisible
    # ancestor_ids = $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    
end

"""
Overload Agents.jl's agent2string function with custom fields for Life

For more information see: https://juliadynamics.github.io/Agents.jl/stable/examples/agents_visualizations/#Agent-inspection
https://stackoverflow.com/questions/37031133/how-do-you-format-a-string-when-interpolated-in-julia
"""
function Agents.agent2string(agent::Life)
    """
    Life
    id = $(agent.id)
    pos = ($(join([@sprintf("%.2f", i) for i in agent.pos],", ")))
    vel = ($(join([@sprintf("%.2f", i) for i in agent.vel],", ")))
    parentplanet (†): $(agent.parentplanet.id)
    composition = [$(join([@sprintf("%.2f", i) for i in agent.composition],", "))]
    destination (†): $(agent.destination.id)
    destination_distance: $(agent.destination_distance)
    ancestors (†): $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    departure_timestep: $(@sprintf("%.2f", agent.departure_timestep))
    arrival_timestep: $(@sprintf("%.2f", agent.arrival_timestep))
    is_mission: $(agent.is_mission)
    """
    ## Have to exclude this because it's taking up making the rest of the screen invisible
    # ancestor_ids = $(length(agent.ancestors) == 0 ? "No ancestors" : [i.id for i in agent.ancestors])
    
end

end # module

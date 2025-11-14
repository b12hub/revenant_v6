# /agents/b_series/arvr_agent.py
from core.agent_base import RevenantAgentBase
import asyncio
from typing import Dict, List, Any
import json
import math


class ARVRAgent(RevenantAgentBase):
    """Build 3D scene configurations for AR/VR systems using spatial computing concepts."""
    # Agent metadata for registry and discovery
    metadata = {
        "name": "ARVRAgent",
        "version": "1.0.0",
        "series": "b_series",
        "description": "Creates and configures 3D scenes for augmented and virtual reality applications",
        "module": "agents.b_series.arvr_agent"
    }
    def __init__(self):
        super().__init__(name=self.metadata["name"],
            description=self.metadata["description"]
        )

        self.scene_templates = {}
        self.object_library = {}

    async def setup(self):
        # Initialize scene templates
        self.scene_templates = {
            "office": {
                "description": "Modern office environment",
                "objects": ["desk", "chair", "computer", "bookshelf"],
                "lighting": "natural",
                "scale": "real_world"
            },
            "classroom": {
                "description": "Educational classroom setting",
                "objects": ["whiteboard", "desks", "chairs", "projector"],
                "lighting": "fluorescent",
                "scale": "real_world"
            },
            "showroom": {
                "description": "Product展示room environment",
                "objects": ["display_stands", "lighting_rigs", "product_models"],
                "lighting": "studio",
                "scale": "exhibition"
            },
            "outdoor": {
                "description": "Outdoor natural environment",
                "objects": ["trees", "terrain", "skybox", "natural_light"],
                "lighting": "daylight",
                "scale": "large_scale"
            }
        }

        # Initialize 3D object library
        self.object_library = {
            "furniture": ["chair", "desk", "table", "sofa", "bookshelf", "cabinet"],
            "electronics": ["computer", "monitor", "printer", "projector", "tv"],
            "decorations": ["plant", "painting", "lamp", "rug", "clock"],
            "structural": ["wall", "floor", "ceiling", "door", "window"],
            "interactive": ["button", "slider", "toggle", "menu", "hud"]
        }

        await asyncio.sleep(0.1)

    async def run(self, input_data: dict):
        try:
            # Validate input
            scene_name = input_data.get("scene_name", "default")
            scene_type = input_data.get("scene_type", "office")
            requirements = input_data.get("requirements", {})

            # Generate scene configuration
            scene_config = await self._create_scene_configuration(scene_name, scene_type, requirements)

            # Optimize scene for target platform
            optimization = await self._optimize_scene(scene_config, requirements)

            # Generate interaction design
            interaction_design = await self._design_interactions(scene_config, requirements)

            # Create deployment package
            deployment_package = await self._create_deployment_package(scene_config, optimization)

            result = {
                "scene_configuration": scene_config,
                "optimization_results": optimization,
                "interaction_design": interaction_design,
                "deployment_ready": deployment_package,
                "performance_metrics": await self._calculate_performance_metrics(scene_config, optimization),
                "compatibility_check": await self._check_platform_compatibility(scene_config, requirements)
            }

            return {
                "agent": self.name,
                "status": "ok",
                "summary": f"AR/VR scene '{scene_name}' configured with {len(scene_config['objects'])} objects, optimized for {optimization['target_platform']}",
                "data": result
            }

        except Exception as e:
            return await self.on_error(str(e))

    async def _create_scene_configuration(self, scene_name: str, scene_type: str, requirements: Dict[str, Any]) -> Dict[
        str, Any]:
        """Create comprehensive 3D scene configuration"""
        template = self.scene_templates.get(scene_type, self.scene_templates["office"])

        # Generate scene objects based on template and requirements
        scene_objects = await self._generate_scene_objects(template, requirements)

        # Configure lighting
        lighting_config = await self._configure_lighting(template, requirements)

        # Set up spatial audio
        audio_config = await self._configure_audio(scene_type, requirements)

        # Define camera and navigation
        camera_config = await self._configure_camera(scene_type, requirements)

        return {
            "scene_metadata": {
                "name": scene_name,
                "type": scene_type,
                "description": template["description"],
                "version": "1.0",
                "created_date": await self._get_current_timestamp(),
                "author": "ARVRAgent"
            },
            "environment": {
                "lighting": lighting_config,
                "audio": audio_config,
                "background": await self._configure_background(scene_type),
                "ambient_effects": await self._configure_ambient_effects(scene_type)
            },
            "objects": scene_objects,
            "camera": camera_config,
            "navigation": await self._configure_navigation(scene_type, scene_objects),
            "physics": await self._configure_physics(scene_type),
            "rendering": await self._configure_rendering(requirements)
        }

    async def _generate_scene_objects(self, template: Dict[str, Any], requirements: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Generate 3D objects for the scene"""
        base_objects = template.get("objects", [])
        custom_objects = requirements.get("custom_objects", [])
        all_objects = base_objects + custom_objects

        scene_objects = []
        object_id = 0

        for obj_type in all_objects:
            object_config = await self._create_object_configuration(obj_type, object_id, requirements)
            scene_objects.append(object_config)
            object_id += 1

        # Add required objects from requirements
        required_objects = requirements.get("required_objects", [])
        for req_obj in required_objects:
            if req_obj not in all_objects:
                object_config = await self._create_object_configuration(req_obj, object_id, requirements)
                scene_objects.append(object_config)
                object_id += 1

        return scene_objects

    async def _create_object_configuration(self, obj_type: str, obj_id: int, requirements: Dict[str, Any]) -> Dict[
        str, Any]:
        """Create configuration for a single 3D object"""
        # Determine object category
        category = await self._categorize_object(obj_type)

        # Generate position (simple grid layout for demo)
        position = await self._calculate_object_position(obj_id, len(self.object_library.get(category, [])))

        # Generate scale based on object type
        scale = await self._calculate_object_scale(obj_type)

        # Generate material properties
        material = await self._generate_material_properties(obj_type, category)

        return {
            "id": f"object_{obj_id}",
            "type": obj_type,
            "category": category,
            "position": position,
            "rotation": await self._generate_rotation(obj_type),
            "scale": scale,
            "material": material,
            "collision": await self._configure_collision(obj_type, category),
            "interactivity": await self._configure_interactivity(obj_type, requirements),
            "lod_levels": await self._generate_lod_levels(obj_type),
            "animations": await self._generate_animations(obj_type)
        }

    async def _configure_lighting(self, template: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure scene lighting"""
        lighting_type = template.get("lighting", "natural")
        user_lighting = requirements.get("lighting", {})

        lighting_configs = {
            "natural": {
                "type": "directional",
                "intensity": 1.0,
                "color": "#FFFFFF",
                "position": {"x": 50, "y": 100, "z": 50},
                "shadows": True,
                "ambient_light": {"color": "#87CEEB", "intensity": 0.3}
            },
            "fluorescent": {
                "type": "point",
                "intensity": 0.8,
                "color": "#F5F5DC",
                "position": {"x": 0, "y": 10, "z": 0},
                "shadows": False,
                "ambient_light": {"color": "#FFFFFF", "intensity": 0.4}
            },
            "studio": {
                "type": "spot",
                "intensity": 1.2,
                "color": "#FFFFFF",
                "position": {"x": 0, "y": 15, "z": 20},
                "shadows": True,
                "ambient_light": {"color": "#333333", "intensity": 0.1}
            },
            "daylight": {
                "type": "directional",
                "intensity": 1.5,
                "color": "#FFEBB7",
                "position": {"x": 100, "y": 150, "z": 100},
                "shadows": True,
                "ambient_light": {"color": "#87CEEB", "intensity": 0.5}
            }
        }

        base_config = lighting_configs.get(lighting_type, lighting_configs["natural"])

        # Apply user overrides
        if user_lighting:
            base_config.update(user_lighting)

        return base_config

    async def _configure_audio(self, scene_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure spatial audio for the scene"""
        audio_presets = {
            "office": {
                "background": "office_ambience",
                "volume": 0.3,
                "spatial": True,
                "reverb": "medium_room"
            },
            "classroom": {
                "background": "classroom_ambience",
                "volume": 0.4,
                "spatial": True,
                "reverb": "large_room"
            },
            "showroom": {
                "background": "showroom_ambience",
                "volume": 0.2,
                "spatial": True,
                "reverb": "large_hall"
            },
            "outdoor": {
                "background": "nature_ambience",
                "volume": 0.5,
                "spatial": True,
                "reverb": "outdoor"
            }
        }

        audio_config = audio_presets.get(scene_type, audio_presets["office"])

        # Apply user audio requirements
        user_audio = requirements.get("audio", {})
        if user_audio:
            audio_config.update(user_audio)

        return audio_config

    async def _configure_camera(self, scene_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure camera settings for the scene"""
        camera_presets = {
            "office": {
                "position": {"x": 0, "y": 1.7, "z": 5},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "fov": 60,
                "near": 0.1,
                "far": 1000,
                "type": "perspective"
            },
            "classroom": {
                "position": {"x": 0, "y": 1.7, "z": 8},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "fov": 70,
                "near": 0.1,
                "far": 1000,
                "type": "perspective"
            },
            "showroom": {
                "position": {"x": 0, "y": 1.5, "z": 10},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "fov": 50,
                "near": 0.1,
                "far": 500,
                "type": "perspective"
            },
            "outdoor": {
                "position": {"x": 0, "y": 1.7, "z": 15},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "fov": 80,
                "near": 0.1,
                "far": 2000,
                "type": "perspective"
            }
        }

        camera_config = camera_presets.get(scene_type, camera_presets["office"])

        # Apply user camera requirements
        user_camera = requirements.get("camera", {})
        if user_camera:
            camera_config.update(user_camera)

        return camera_config

    async def _configure_background(self, scene_type: str) -> Dict[str, Any]:
        """Configure scene background"""
        backgrounds = {
            "office": {
                "type": "color",
                "value": "#87CEEB",
                "skybox": False
            },
            "classroom": {
                "type": "color",
                "value": "#FFFFFF",
                "skybox": False
            },
            "showroom": {
                "type": "gradient",
                "value": ["#1a2a6c", "#b21f1f", "#fdbb2d"],
                "skybox": True
            },
            "outdoor": {
                "type": "skybox",
                "value": "daylight_sky",
                "skybox": True
            }
        }

        return backgrounds.get(scene_type, backgrounds["office"])

    async def _configure_ambient_effects(self, scene_type: str) -> List[Dict[str, Any]]:
        """Configure ambient effects for the scene"""
        effects_presets = {
            "office": [
                {"type": "particles", "effect": "dust_motes", "intensity": 0.1},
                {"type": "fog", "density": 0.01, "color": "#FFFFFF"}
            ],
            "classroom": [
                {"type": "particles", "effect": "chalk_dust", "intensity": 0.05},
                {"type": "fog", "density": 0.005, "color": "#F5F5F5"}
            ],
            "showroom": [
                {"type": "light_rays", "intensity": 0.3, "color": "#FFFFFF"},
                {"type": "bloom", "threshold": 0.7, "strength": 0.5}
            ],
            "outdoor": [
                {"type": "fog", "density": 0.02, "color": "#87CEEB"},
                {"type": "wind", "strength": 0.3, "direction": {"x": 1, "y": 0, "z": 0}}
            ]
        }

        return effects_presets.get(scene_type, [])

    async def _configure_navigation(self, scene_type: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure navigation and movement within the scene"""
        navigation_presets = {
            "office": {
                "type": "teleport",
                "movement_speed": 2.0,
                "collision_detection": True,
                "boundaries": await self._calculate_navigation_boundaries(objects),
                "waypoints": await self._generate_waypoints(objects, "office")
            },
            "classroom": {
                "type": "smooth",
                "movement_speed": 1.5,
                "collision_detection": True,
                "boundaries": await self._calculate_navigation_boundaries(objects),
                "waypoints": await self._generate_waypoints(objects, "classroom")
            },
            "showroom": {
                "type": "guided",
                "movement_speed": 1.0,
                "collision_detection": True,
                "boundaries": await self._calculate_navigation_boundaries(objects),
                "waypoints": await self._generate_waypoints(objects, "showroom")
            },
            "outdoor": {
                "type": "free",
                "movement_speed": 3.0,
                "collision_detection": False,
                "boundaries": {"min": {"x": -100, "y": 0, "z": -100}, "max": {"x": 100, "y": 50, "z": 100}},
                "waypoints": await self._generate_waypoints(objects, "outdoor")
            }
        }

        return navigation_presets.get(scene_type, navigation_presets["office"])

    async def _configure_physics(self, scene_type: str) -> Dict[str, Any]:
        """Configure physics settings for the scene"""
        physics_presets = {
            "office": {
                "gravity": -9.81,
                "friction": 0.5,
                "restitution": 0.3,
                "collision_layers": ["static", "dynamic", "kinematic"]
            },
            "classroom": {
                "gravity": -9.81,
                "friction": 0.6,
                "restitution": 0.2,
                "collision_layers": ["static", "dynamic"]
            },
            "showroom": {
                "gravity": -9.81,
                "friction": 0.8,
                "restitution": 0.1,
                "collision_layers": ["static"]
            },
            "outdoor": {
                "gravity": -9.81,
                "friction": 0.3,
                "restitution": 0.5,
                "collision_layers": ["terrain", "dynamic"]
            }
        }

        return physics_presets.get(scene_type, physics_presets["office"])

    async def _configure_rendering(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure rendering settings"""
        quality_preset = requirements.get("quality", "medium")

        rendering_presets = {
            "low": {
                "shadow_quality": "low",
                "texture_quality": "low",
                "anti_aliasing": "none",
                "post_processing": False,
                "max_lights": 2
            },
            "medium": {
                "shadow_quality": "medium",
                "texture_quality": "medium",
                "anti_aliasing": "fxaa",
                "post_processing": True,
                "max_lights": 4
            },
            "high": {
                "shadow_quality": "high",
                "texture_quality": "high",
                "anti_aliasing": "msaa",
                "post_processing": True,
                "max_lights": 8
            },
            "ultra": {
                "shadow_quality": "ultra",
                "texture_quality": "ultra",
                "anti_aliasing": "msaa_4x",
                "post_processing": True,
                "max_lights": 16
            }
        }

        return rendering_presets.get(quality_preset, rendering_presets["medium"])

    async def _optimize_scene(self, scene_config: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize scene for target platform and performance"""
        target_platform = requirements.get("platform", "standalone_vr")

        optimization_strategies = {
            "mobile_ar": {
                "target_platform": "mobile_ar",
                "polygon_reduction": 0.7,
                "texture_compression": "high",
                "lod_distance_multiplier": 0.5,
                "light_count_limit": 2,
                "shadow_quality": "low"
            },
            "standalone_vr": {
                "target_platform": "standalone_vr",
                "polygon_reduction": 0.3,
                "texture_compression": "medium",
                "lod_distance_multiplier": 0.8,
                "light_count_limit": 4,
                "shadow_quality": "medium"
            },
            "pc_vr": {
                "target_platform": "pc_vr",
                "polygon_reduction": 0.1,
                "texture_compression": "low",
                "lod_distance_multiplier": 1.0,
                "light_count_limit": 8,
                "shadow_quality": "high"
            },
            "web_ar": {
                "target_platform": "web_ar",
                "polygon_reduction": 0.8,
                "texture_compression": "very_high",
                "lod_distance_multiplier": 0.3,
                "light_count_limit": 1,
                "shadow_quality": "none"
            }
        }

        optimization = optimization_strategies.get(target_platform, optimization_strategies["standalone_vr"])

        # Apply optimizations to scene config
        optimized_scene = await self._apply_optimizations(scene_config, optimization)

        return {
            "optimization_strategy": optimization,
            "optimized_elements": optimized_scene,
            "performance_gain": await self._calculate_performance_gain(scene_config, optimized_scene),
            "compatibility_notes": await self._generate_compatibility_notes(optimization)
        }

    async def _design_interactions(self, scene_config: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design user interactions for the scene"""
        interaction_types = requirements.get("interaction_types", ["grab", "point", "select"])

        interactions = {
            "grab": {
                "enabled": "grab" in interaction_types,
                "physics_based": True,
                "grab_distance": 2.0,
                "throw_force": 5.0
            },
            "point": {
                "enabled": "point" in interaction_types,
                "ray_length": 10.0,
                "hover_feedback": True,
                "selection_highlight": True
            },
            "select": {
                "enabled": "select" in interaction_types,
                "method": "raycast",
                "confirmation_required": False,
                "deselect_on_lose_focus": True
            },
            "ui_interactions": {
                "enabled": any(t in interaction_types for t in ["button", "slider", "menu"]),
                "ui_elements": await self._generate_ui_elements(scene_config, requirements)
            }
        }

        # Configure object-specific interactions
        object_interactions = await self._configure_object_interactions(scene_config["objects"], requirements)

        return {
            "interaction_system": interactions,
            "object_interactions": object_interactions,
            "gesture_support": await self._configure_gestures(requirements),
            "accessibility": await self._configure_accessibility(requirements)
        }

    async def _create_deployment_package(self, scene_config: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[
        str, Any]:
        """Create deployment-ready package for the scene"""
        target_platform = optimization["optimization_strategy"]["target_platform"]

        deployment_configs = {
            "mobile_ar": {
                "format": "usdz",
                "compression": "high",
                "file_size_limit": 50,  # MB
                "required_features": ["arkit", "arcore"],
                "fallback_strategy": "web_ar"
            },
            "standalone_vr": {
                "format": "gltf",
                "compression": "medium",
                "file_size_limit": 200,  # MB
                "required_features": ["vr_controllers", "spatial_audio"],
                "fallback_strategy": "pc_desktop"
            },
            "pc_vr": {
                "format": "gltf",
                "compression": "low",
                "file_size_limit": 500,  # MB
                "required_features": ["high_res_textures", "advanced_lighting"],
                "fallback_strategy": "standalone_vr"
            },
            "web_ar": {
                "format": "gltf",
                "compression": "very_high",
                "file_size_limit": 10,  # MB
                "required_features": ["webgl2", "webar"],
                "fallback_strategy": "static_images"
            }
        }

        deployment = deployment_configs.get(target_platform, deployment_configs["standalone_vr"])

        return {
            "deployment_config": deployment,
            "build_instructions": await self._generate_build_instructions(scene_config, optimization),
            "testing_checklist": await self._generate_testing_checklist(target_platform),
            "performance_budget": await self._calculate_performance_budget(scene_config, deployment)
        }

    # Helper methods with mock implementations
    async def _get_current_timestamp(self) -> str:
        """Get current timestamp for scene metadata"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def _categorize_object(self, obj_type: str) -> str:
        """Categorize object into predefined categories"""
        for category, objects in self.object_library.items():
            if obj_type in objects:
                return category
        return "misc"

    async def _calculate_object_position(self, obj_id: int, total_objects: int) -> Dict[str, float]:
        """Calculate object position in scene (grid layout)"""
        grid_size = math.ceil(math.sqrt(total_objects + 1))
        row = obj_id // grid_size
        col = obj_id % grid_size

        return {
            "x": (col - grid_size / 2) * 3.0,
            "y": 0,
            "z": (row - grid_size / 2) * 3.0
        }

    async def _calculate_object_scale(self, obj_type: str) -> Dict[str, float]:
        """Calculate appropriate scale for object type"""
        scale_presets = {
            "chair": {"x": 1, "y": 1, "z": 1},
            "desk": {"x": 2, "y": 1, "z": 1},
            "computer": {"x": 0.5, "y": 0.5, "z": 0.5},
            "bookshelf": {"x": 1.5, "y": 2, "z": 0.5},
            "whiteboard": {"x": 2, "y": 1.5, "z": 0.1}
        }

        return scale_presets.get(obj_type, {"x": 1, "y": 1, "z": 1})

    async def _generate_material_properties(self, obj_type: str, category: str) -> Dict[str, Any]:
        """Generate material properties for object"""
        material_presets = {
            "furniture": {
                "type": "standard",
                "color": "#8B4513",
                "roughness": 0.7,
                "metalness": 0.1,
                "emissive": "#000000"
            },
            "electronics": {
                "type": "standard",
                "color": "#2F4F4F",
                "roughness": 0.3,
                "metalness": 0.8,
                "emissive": "#1a1a1a"
            },
            "decorations": {
                "type": "standard",
                "color": "#FFFFFF",
                "roughness": 0.5,
                "metalness": 0.2,
                "emissive": "#000000"
            }
        }

        return material_presets.get(category, {
            "type": "standard",
            "color": "#808080",
            "roughness": 0.5,
            "metalness": 0.5,
            "emissive": "#000000"
        })

    async def _generate_rotation(self, obj_type: str) -> Dict[str, float]:
        """Generate object rotation"""
        return {"x": 0, "y": 0, "z": 0}

    async def _configure_collision(self, obj_type: str, category: str) -> Dict[str, Any]:
        """Configure collision properties for object"""
        return {
            "enabled": category in ["furniture", "structural"],
            "type": "box" if category in ["furniture", "electronics"] else "mesh",
            "is_trigger": category == "interactive"
        }

    async def _configure_interactivity(self, obj_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure interactivity for object"""
        interactive_objects = requirements.get("interactive_objects", [])

        return {
            "interactive": obj_type in interactive_objects,
            "interaction_type": "grab" if obj_type in ["chair", "desk"] else "select",
            "feedback_type": "haptic" if "grab" in requirements.get("interaction_types", []) else "visual"
        }

    async def _generate_lod_levels(self, obj_type: str) -> List[Dict[str, Any]]:
        """Generate Level of Detail (LOD) configurations"""
        return [
            {"distance": 0, "quality": "high"},
            {"distance": 10, "quality": "medium"},
            {"distance": 20, "quality": "low"}
        ]

    async def _generate_animations(self, obj_type: str) -> List[Dict[str, Any]]:
        """Generate animation configurations for object"""
        animations = []

        if obj_type in ["door", "cabinet"]:
            animations.append({
                "name": "open_close",
                "type": "rotation",
                "target": {"x": 0, "y": 90, "z": 0},
                "duration": 1.0
            })

        return animations

    async def _calculate_navigation_boundaries(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate navigation boundaries based on object positions"""
        if not objects:
            return {"min": {"x": -10, "y": 0, "z": -10}, "max": {"x": 10, "y": 3, "z": 10}}

        x_positions = [obj["position"]["x"] for obj in objects]
        z_positions = [obj["position"]["z"] for obj in objects]

        return {
            "min": {
                "x": min(x_positions) - 2,
                "y": 0,
                "z": min(z_positions) - 2
            },
            "max": {
                "x": max(x_positions) + 2,
                "y": 3,
                "z": max(z_positions) + 2
            }
        }

    async def _generate_waypoints(self, objects: List[Dict[str, Any]], scene_type: str) -> List[Dict[str, Any]]:
        """Generate navigation waypoints"""
        waypoints = []

        # Add waypoints near interactive objects
        interactive_objects = [obj for obj in objects if obj.get("interactivity", {}).get("interactive", False)]

        for i, obj in enumerate(interactive_objects[:3]):  # Max 3 waypoints
            waypoints.append({
                "id": f"waypoint_{i}",
                "position": {
                    "x": obj["position"]["x"],
                    "y": obj["position"]["y"],
                    "z": obj["position"]["z"] + 2
                },
                "description": f"Near {obj['type']}",
                "auto_rotate": True
            })

        return waypoints

    async def _apply_optimizations(self, scene_config: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations to scene configuration"""
        optimized = scene_config.copy()

        # Reduce object count if needed
        if optimization["polygon_reduction"] < 0.5:
            # Keep only essential objects
            essential_objects = [obj for obj in scene_config["objects"] if
                                 obj["category"] in ["furniture", "structural"]]
            optimized["objects"] = essential_objects[:5]  # Limit to 5 objects

        # Adjust rendering quality
        optimized["rendering"].update({
            "shadow_quality": optimization["shadow_quality"],
            "max_lights": optimization["light_count_limit"]
        })

        return optimized

    async def _calculate_performance_gain(self, original: Dict[str, Any], optimized: Dict[str, Any]) -> float:
        """Calculate performance gain from optimization"""
        original_objects = len(original["objects"])
        optimized_objects = len(optimized["objects"])

        if original_objects == 0:
            return 0.0

        object_reduction = (original_objects - optimized_objects) / original_objects
        return min(0.8, object_reduction * 1.5)  # Cap at 80% improvement

    async def _generate_compatibility_notes(self, optimization: Dict[str, Any]) -> List[str]:
        """Generate compatibility notes for target platform"""
        platform = optimization["target_platform"]

        notes = {
            "mobile_ar": [
                "Optimized for iOS ARKit and Android ARCore",
                "Limited to 50MB total scene size",
                "Recommend using simple materials and low-poly models"
            ],
            "standalone_vr": [
                "Compatible with Oculus Quest and similar standalone VR devices",
                "Maintain 72fps for comfortable VR experience",
                "Use baked lighting where possible"
            ],
            "pc_vr": [
                "Optimized for high-end PC VR systems",
                "Can use real-time lighting and higher resolution textures",
                "Target 90fps for smooth experience"
            ],
            "web_ar": [
                "Compatible with WebXR-enabled browsers",
                "Very strict performance budget",
                "Use compressed textures and simple geometries"
            ]
        }

        return notes.get(platform, ["No specific compatibility notes"])

    async def _generate_ui_elements(self, scene_config: Dict[str, Any], requirements: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Generate UI elements for the scene"""
        ui_elements = []

        # Add basic UI elements based on requirements
        if requirements.get("show_instructions", True):
            ui_elements.append({
                "type": "instruction_panel",
                "position": {"x": 0, "y": 2, "z": 3},
                "content": "Welcome to the AR/VR experience",
                "size": {"width": 2, "height": 1}
            })

        if requirements.get("show_controls", True):
            ui_elements.append({
                "type": "control_hint",
                "position": {"x": -1, "y": 1.5, "z": 2},
                "content": "Use controllers to interact",
                "size": {"width": 1, "height": 0.5}
            })

        return ui_elements

    async def _configure_object_interactions(self, objects: List[Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[
        str, Any]:
        """Configure specific interactions for each object"""
        interactions = {}

        for obj in objects:
            if obj.get("interactivity", {}).get("interactive", False):
                interactions[obj["id"]] = {
                    "available_actions": ["select", "hover"],
                    "feedback_types": ["visual", "haptic"],
                    "custom_events": await self._generate_custom_events(obj["type"])
                }

        return interactions

    async def _configure_gestures(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure gesture recognition support"""
        return {
            "hand_tracking": requirements.get("hand_tracking", False),
            "supported_gestures": ["point", "grab", "pinch"] if requirements.get("hand_tracking", False) else [],
            "confidence_threshold": 0.7
        }

    async def _configure_accessibility(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure accessibility features"""
        return {
            "voice_commands": requirements.get("voice_commands", False),
            "subtitles": requirements.get("subtitles", True),
            "color_blind_mode": requirements.get("color_blind_mode", False),
            "motion_sickness_reduction": requirements.get("reduce_motion", False)
        }

    async def _generate_custom_events(self, obj_type: str) -> List[str]:
        """Generate custom events for object interactions"""
        event_templates = {
            "computer": ["on_startup", "on_shutdown", "on_interact"],
            "door": ["on_open", "on_close", "on_lock"],
            "button": ["on_press", "on_release", "on_hover"]
        }

        return event_templates.get(obj_type, ["on_interact", "on_hover"])

    async def _generate_build_instructions(self, scene_config: Dict[str, Any], optimization: Dict[str, Any]) -> List[
        str]:
        """Generate build instructions for deployment"""
        platform = optimization["optimization_strategy"]["target_platform"]

        instructions = {
            "mobile_ar": [
                "Export scene as USDZ format",
                "Compress textures to JPEG/PNG format",
                "Test on target iOS/Android devices",
                "Verify AR plane detection works correctly"
            ],
            "standalone_vr": [
                "Build for Android target platform",
                "Enable Oculus integration",
                "Optimize for 72fps performance",
                "Test controller interactions thoroughly"
            ],
            "pc_vr": [
                "Build for Windows target platform",
                "Enable SteamVR/OpenXR support",
                "Implement advanced rendering features",
                "Test with various PC VR headsets"
            ],
            "web_ar": [
                "Export as GLTF with Draco compression",
                "Implement WebXR device API",
                "Optimize for mobile browser performance",
                "Test cross-browser compatibility"
            ]
        }

        return instructions.get(platform, [
            "Build scene for target platform",
            "Test all interactions and performance",
            "Verify compatibility with target devices"
        ])

    async def _generate_testing_checklist(self, target_platform: str) -> List[str]:
        """Generate testing checklist for the scene"""
        checklists = {
            "mobile_ar": [
                "Test on iOS ARKit compatible devices",
                "Test on Android ARCore compatible devices",
                "Verify plane detection and surface tracking",
                "Check performance on mid-range devices"
            ],
            "standalone_vr": [
                "Test on Oculus Quest devices",
                "Verify 72fps performance target",
                "Test controller haptics and inputs",
                "Check for motion sickness triggers"
            ],
            "pc_vr": [
                "Test on various PC VR headsets",
                "Verify 90fps performance target",
                "Test advanced graphics features",
                "Check GPU memory usage"
            ],
            "web_ar": [
                "Test on WebXR compatible browsers",
                "Verify mobile performance",
                "Check cross-browser functionality",
                "Test fallback scenarios"
            ]
        }

        return checklists.get(target_platform, [
            "Test basic functionality",
            "Verify performance metrics",
            "Check user experience quality"
        ])

    async def _calculate_performance_budget(self, scene_config: Dict[str, Any], deployment: Dict[str, Any]) -> Dict[
        str, Any]:
        """Calculate performance budget for the scene"""
        object_count = len(scene_config["objects"])
        texture_count = sum(1 for obj in scene_config["objects"] if obj.get("material", {}).get("texture"))

        return {
            "target_fps": 90 if deployment["target_platform"] == "pc_vr" else 72,
            "max_polygons": 100000 if deployment["target_platform"] == "pc_vr" else 50000,
            "max_textures": 50,
            "memory_budget_mb": deployment["file_size_limit"],
            "current_usage": {
                "polygons": object_count * 1000,  # Rough estimate
                "textures": texture_count,
                "memory_mb": object_count * 2  # Rough estimate
            }
        }

    async def _calculate_performance_metrics(self, scene_config: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[
        str, Any]:
        """Calculate performance metrics for the scene"""
        object_count = len(scene_config["objects"])
        light_count = 1  # Base lighting

        # Estimate performance score (0-100)
        base_score = 100
        score_reduction = object_count * 0.5 + light_count * 2

        performance_score = max(0, base_score - score_reduction)

        return {
            "estimated_performance_score": performance_score,
            "recommended_platforms": await self._recommend_platforms(performance_score),
            "bottlenecks": await self._identify_performance_bottlenecks(scene_config),
            "optimization_suggestions": await self._suggest_performance_improvements(performance_score)
        }

    async def _check_platform_compatibility(self, scene_config: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[
        str, Any]:
        """Check compatibility with target platforms"""
        target_platform = requirements.get("platform", "standalone_vr")

        compatibility_matrix = {
            "mobile_ar": {
                "compatible": len(scene_config["objects"]) <= 20,
                "issues": ["High object count may affect performance"] if len(scene_config["objects"]) > 20 else [],
                "recommendations": ["Reduce object count", "Use simpler materials"] if len(
                    scene_config["objects"]) > 20 else ["Scene is well optimized"]
            },
            "standalone_vr": {
                "compatible": len(scene_config["objects"]) <= 50,
                "issues": [],
                "recommendations": ["Scene is VR-ready"]
            },
            "pc_vr": {
                "compatible": True,
                "issues": [],
                "recommendations": ["Can add more complex effects"]
            },
            "web_ar": {
                "compatible": len(scene_config["objects"]) <= 10,
                "issues": ["Object count too high for web"] if len(scene_config["objects"]) > 10 else [],
                "recommendations": ["Significantly reduce complexity"] if len(scene_config["objects"]) > 10 else [
                    "Good for web deployment"]
            }
        }

        return compatibility_matrix.get(target_platform, {
            "compatible": True,
            "issues": [],
            "recommendations": ["No specific compatibility issues detected"]
        })

    async def _recommend_platforms(self, performance_score: float) -> List[str]:
        """Recommend suitable platforms based on performance score"""
        if performance_score >= 80:
            return ["pc_vr", "standalone_vr", "mobile_ar", "web_ar"]
        elif performance_score >= 60:
            return ["standalone_vr", "mobile_ar"]
        elif performance_score >= 40:
            return ["mobile_ar"]
        else:
            return ["web_ar"]  # Lowest requirements

    async def _identify_performance_bottlenecks(self, scene_config: Dict[str, Any]) -> List[str]:
        """Identify potential performance bottlenecks"""
        bottlenecks = []

        object_count = len(scene_config["objects"])
        if object_count > 50:
            bottlenecks.append("High object count may impact performance")

        if scene_config["rendering"]["shadow_quality"] == "high":
            bottlenecks.append("High-quality shadows can be performance intensive")

        if any(obj.get("material", {}).get("texture") for obj in scene_config["objects"]):
            bottlenecks.append("Textured materials increase memory usage")

        return bottlenecks

    async def _suggest_performance_improvements(self, performance_score: float) -> List[str]:
        """Suggest performance improvements based on score"""
        if performance_score >= 80:
            return ["Scene is well optimized. No changes needed."]
        elif performance_score >= 60:
            return ["Consider reducing object count", "Use lower resolution textures"]
        elif performance_score >= 40:
            return ["Reduce object count significantly", "Disable shadows", "Use simple materials"]
        else:
            return ["Major optimization required", "Reduce to essential objects only",
                    "Use only color materials without textures"]
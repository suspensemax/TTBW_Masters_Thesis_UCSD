try:
    from csdl.lang.model import Model
except:
    pass
from typing import Dict, Any, Set
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.utils.find_promoted_name import find_promoted_name


def collect_constraints(
    model: 'Model',
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
    namespace: str = '',
    constraints: Dict[str, Dict[str, Any]] = dict(),
) -> Dict[str, Dict[str, Any]]:
    for k, constraint in model.constraints.items():
        name = find_promoted_name(
            k,
            model.promoted_to_unpromoted,
            model.unpromoted_to_promoted,
        )
        name = find_promoted_name(
            prepend_namespace(namespace, name),
            promoted_to_unpromoted,
            unpromoted_to_promoted,
        )

        # TODO: make this more helpful for users to find both
        # constraints
        if name in constraints.keys():
            raise ValueError(
                f"Redundant constraint {name} declared."
            )

        constraints[name] = constraint

    for s in model.subgraphs:
        constraints = collect_constraints(
            s.submodel,
            promoted_to_unpromoted,
            unpromoted_to_promoted,
            prepend_namespace(namespace, s.name),
            constraints=constraints,
        )

    return constraints

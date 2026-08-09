"""``CallCollector``, split across one module per concern.

The class is one object -- every method reads and writes the same ``self`` --
but its 2,400 lines are assembled here from mixins so that each concern is a
file you can hold in your head. See ``modularisation_plan.md`` in the parent
package for the layout, the invariants that keep the MRO inert, and what this
split deliberately does *not* attempt.

``CallCollector`` is the only public name.
"""

from .collector import CallCollector

__all__ = ["CallCollector"]

Limitations that are awared:
(1) It can produce multiple targets when runtime configuration would choose only one:
HANDLERS[name]()
It may miss callable identities when they are produced through unsupported dynamic behavior, such as:
reflection
runtime monkey-patching
arbitrary descriptor logic
callable collection comprehensions
some collection mutations
external factories whose source is not analyzed
complex aliasing
values loaded dynamically from configuration or plugins
It also deliberately trusts only resolved project callable IDs when propagating function identities. This lowers false positives but means external callback behavior may remain unresolved.

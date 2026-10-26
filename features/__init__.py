"""Feature helpers shared by the Streamlit pages and the tests.

The training datasets store every feature as it stood *before* each race or
match (rolling windows are shifted by one, Elo is saved before the update).
That is right for training, but the apps predict the *next* event, so they need
the same features *after* the most recent one. The ``build_current_state``
functions in this package replay the feature-generation notebooks over the
full history and return that up-to-date state.
"""

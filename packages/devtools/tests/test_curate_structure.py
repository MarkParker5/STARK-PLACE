from stark_devtools.profiler.curate import curate
from stark_devtools.profiler.structure import build_graph, extract_classes


async def test_curate_produces_ordered_steps(run_utterance):
    events = await run_utterance("play Metallica and turn off the kitchen lights")
    steps = curate(events)

    labels = [s.label for s in steps]
    assert "process" in labels
    assert "search" in labels
    assert "pattern match" in labels
    assert labels.count("dispatch") == 2  # two commands dispatched

    # steps are ordered and carry input+output
    assert [s.seq for s in steps] == sorted(s.seq for s in steps)
    process = next(s for s in steps if s.label == "process")
    assert process.input["string"] == "play Metallica and turn off the kitchen lights"
    assert len(process.output["results"]) == 2


def test_structure_static_graph_no_imports():
    # AST-only extraction: works even though optional deps (vosk/torch/spacy) aren't installed.
    classes = extract_classes()
    assert "CommandsContext" in classes
    assert "PatternParser" in classes
    cc = classes["CommandsContext"]
    assert cc.public > 0
    graph = build_graph()  # static only
    assert graph["meta"]["class_count"] == len(classes)
    assert graph["meta"]["traced"] is False


async def test_structure_weights_reflect_calls(run_utterance):
    events = await run_utterance("play Metallica")
    graph = build_graph(events)
    assert graph["meta"]["traced"] is True

    by_id = {n["id"]: n for n in graph["nodes"]}
    # PatternParser is called many times during a match → active with call weight
    pp = by_id["PatternParser"]
    assert pp["active"] and pp["calls"] > 0
    # weight formula: calls + 3*public + private + 2*relations
    expected = pp["calls"] + 3 * pp["public"] + pp["private"] + 2 * pp["relations"]
    assert abs(pp["weight"] - expected) < 1e-6
    # nodes are sorted heaviest-first
    weights = [n["weight"] for n in graph["nodes"]]
    assert weights == sorted(weights, reverse=True)

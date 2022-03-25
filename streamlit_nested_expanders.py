from typing import Optional, Iterable

import streamlit as st
from streamlit import cursor, caching
from streamlit import legacy_caching
from streamlit import type_util
from streamlit import util
from streamlit.cursor import Cursor
from streamlit.script_run_context import get_script_run_ctx
from streamlit.errors import StreamlitAPIException
from streamlit.errors import NoSessionContext
from streamlit.proto import Block_pb2
from streamlit.proto import ForwardMsg_pb2
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.logger import get_logger

from streamlit.elements.balloons import BalloonsMixin
from streamlit.elements.button import ButtonMixin
from streamlit.elements.markdown import MarkdownMixin
from streamlit.elements.text import TextMixin
from streamlit.elements.alert import AlertMixin
from streamlit.elements.json import JsonMixin
from streamlit.elements.doc_string import HelpMixin
from streamlit.elements.exception import ExceptionMixin
from streamlit.elements.bokeh_chart import BokehMixin
from streamlit.elements.graphviz_chart import GraphvizMixin
from streamlit.elements.plotly_chart import PlotlyMixin
from streamlit.elements.deck_gl_json_chart import PydeckMixin
from streamlit.elements.map import MapMixin
from streamlit.elements.iframe import IframeMixin
from streamlit.elements.media import MediaMixin
from streamlit.elements.checkbox import CheckboxMixin
from streamlit.elements.multiselect import MultiSelectMixin
from streamlit.elements.metric import MetricMixin
from streamlit.elements.radio import RadioMixin
from streamlit.elements.selectbox import SelectboxMixin
from streamlit.elements.text_widgets import TextWidgetsMixin
from streamlit.elements.time_widgets import TimeWidgetsMixin
from streamlit.elements.progress import ProgressMixin
from streamlit.elements.empty import EmptyMixin
from streamlit.elements.number_input import NumberInputMixin
from streamlit.elements.camera_input import CameraInputMixin
from streamlit.elements.color_picker import ColorPickerMixin
from streamlit.elements.file_uploader import FileUploaderMixin
from streamlit.elements.select_slider import SelectSliderMixin
from streamlit.elements.slider import SliderMixin
from streamlit.elements.snow import SnowMixin
from streamlit.elements.image import ImageMixin
from streamlit.elements.pyplot import PyplotMixin
from streamlit.elements.write import WriteMixin
from streamlit.elements.layouts import LayoutsMixin
from streamlit.elements.form import FormMixin, FormData, current_form_id
from streamlit.state import NoValue

from streamlit.elements.arrow import ArrowMixin
from streamlit.elements.arrow_altair import ArrowAltairMixin
from streamlit.elements.arrow_vega_lite import ArrowVegaLiteMixin
from streamlit.elements.legacy_data_frame import LegacyDataFrameMixin
from streamlit.elements.legacy_altair import LegacyAltairMixin
from streamlit.elements.legacy_vega_lite import LegacyVegaLiteMixin
from streamlit.elements.dataframe_selector import DataFrameSelectorMixin

from streamlit.delta_generator import DeltaGenerator


def _block(self, block_proto=Block_pb2.Block()) -> "DeltaGenerator":
    # Operate on the active DeltaGenerator, in case we're in a `with` block.
    dg = self._active_dg

    # Prevent nested columns & expanders by checking all parents.
    block_type = block_proto.WhichOneof("type")
    # Convert the generator to a list, so we can use it multiple times.
    parent_block_types = frozenset(dg._parent_block_types)
    # if block_type == "column" and block_type in parent_block_types:
    #    raise StreamlitAPIException("Columns may not be nested inside other columns.")
    # if block_type == "expandable" and block_type in parent_block_types:
    #    raise StreamlitAPIException(
    #        "Expanders may not be nested inside other expanders."
    #    )

    if dg._root_container is None or dg._cursor is None:
        return dg

    msg = ForwardMsg_pb2.ForwardMsg()
    msg.metadata.delta_path[:] = dg._cursor.delta_path
    msg.delta.add_block.CopyFrom(block_proto)

    # Normally we'd return a new DeltaGenerator that uses the locked cursor
    # below. But in this case we want to return a DeltaGenerator that uses
    # a brand new cursor for this new block we're creating.
    block_cursor = cursor.RunningCursor(
        root_container=dg._root_container,
        parent_path=dg._cursor.parent_path + (dg._cursor.index,),
    )
    block_dg = DeltaGenerator(
        root_container=dg._root_container,
        cursor=block_cursor,
        parent=dg,
        block_type=block_type,
    )
    # Blocks inherit their parent form ids.
    # NOTE: Container form ids aren't set in proto.
    block_dg._form_data = FormData(current_form_id(dg))

    # Must be called to increment this cursor's index.
    dg._cursor.get_locked_cursor(last_index=None)
    _enqueue_message(msg)

    return block_dg


def _enqueue_message(msg):
    """Enqueues a ForwardMsg proto to send to the app."""
    ctx = get_script_run_ctx()

    if ctx is None:
        raise NoSessionContext()

    ctx.enqueue(msg)


DeltaGenerator._block = _block

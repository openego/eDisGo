# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# It is a fork of the "oedialect" package
# (https://github.com/OpenEnergyPlatform/oedialect), Copyright (c) Reiner
# Lemoine Institut gGmbH and the Open Energy Platform contributors, from which
# the JSON representation of SQL expressions is taken. See LICENSE in this
# directory and the module docstring of __init__.py.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Compiler translating SQLAlchemy expressions into OEP API query documents.

The OEP API takes queries as JSON documents rather than SQL strings, so
:class:`OECompiler` returns a nested ``dict`` where SQLAlchemy's own compiler
returns a string. Only reading is supported - eDisGo never writes to the OEP,
and compiling an INSERT, UPDATE, DELETE or DDL statement raises
:class:`~sqlalchemy.exc.CompileError`.
"""

from sqlalchemy import exc
from sqlalchemy.dialects.postgresql.base import PGCompiler, PGTypeCompiler
from sqlalchemy.sql import elements, operators
from sqlalchemy.sql.compiler import FUNCTIONS, OPERATORS

from edisgo.io.oedialect.api import DEFAULT_SCHEMA


class OECompiler(PGCompiler):
    """
    Compile SELECT statements into the JSON documents of the OEP API.

    The compiled statement is available as the ``string`` attribute, as with
    any SQLAlchemy compiler - it just holds a ``dict`` instead of SQL text.
    Bind parameters compile to callables that the cursor replaces with the
    actual values (see :meth:`bindparam_string`).
    """

    def __str__(self):
        # The compiled statement is a dict, not a string. SQLAlchemy calls
        # str() on it for logging and for the execution context.
        return ""

    def _not_supported(self, statement):
        raise exc.CompileError(
            f"The OEP dialect is read-only, {statement} statements are not supported."
        )

    def visit_insert(self, insert_stmt, **kw):
        self._not_supported("INSERT")

    def visit_update(self, update_stmt, **kw):
        self._not_supported("UPDATE")

    def visit_delete(self, delete_stmt, **kw):
        self._not_supported("DELETE")

    def visit_clauselist(self, clauselist, **kw):
        clauses = [
            s
            for s in (c._compiler_dispatch(self, **kw) for c in clauselist.clauses)
            if s
        ]
        if clauselist.operator in (operators.and_, operators.or_):
            return {
                "type": "operator",
                "operator": OPERATORS[clauselist.operator],
                "operands": clauses,
            }
        return clauses

    def visit_expression_clauselist(self, clauselist, **kw):
        """
        Compile an operator chain, e.g. the clauses of ``and_()`` or ``or_()``.

        SQLAlchemy joins them with the operator as a string; here they become
        the operands of one operator document.
        """
        operator_ = clauselist.operator
        disp = self._get_operator_dispatch(operator_, "expression_clauselist", None)
        if disp:
            return disp(clauselist, operator_, **kw)

        kw["_in_operator_expression"] = True
        operands = [
            c
            for c in (
                clause._compiler_dispatch(self, **kw) for clause in clauselist.clauses
            )
            if c
        ]
        return {
            "type": "operator",
            "operator": OPERATORS[operator_],
            "operands": operands,
        }

    def visit_bindparam(self, bindparam, **kw):
        if bindparam.expanding:
            # SQLAlchemy renders an IN clause as one "expanding" parameter that
            # it substitutes into the SQL string just before execution. The API
            # is sent a document instead of a string, so the values are put into
            # it right away - each wrapped in a callable, like any other bind
            # parameter (see bindparam_string).
            return {
                "type": "grouping",
                "grouping": [
                    (lambda params, value=value: value) for value in bindparam.value
                ],
            }
        return super().visit_bindparam(bindparam, **kw)

    def visit_unary(self, unary, **kw):
        if unary.operator:
            if unary.modifier:
                raise exc.CompileError(
                    "Unary expression does not support operator and modifier "
                    "simultaneously"
                )
            disp = self._get_operator_dispatch(unary.operator, "unary", "operator")
            if disp:
                return disp(unary, unary.operator, **kw)
            return self._generate_generic_unary_operator(
                unary, OPERATORS[unary.operator], **kw
            )
        elif unary.modifier:
            disp = self._get_operator_dispatch(unary.modifier, "unary", "modifier")
            if disp:
                return disp(unary, unary.modifier, **kw)
            return self._generate_generic_unary_modifier(
                unary, OPERATORS[unary.modifier], **kw
            )
        raise exc.CompileError("Unary expression has no operator or modifier")

    def visit_case(self, clause, **kwargs):
        d = {"type": "case"}
        if clause.value is not None:
            d["expression"] = clause.value._compiler_dispatch(self, **kwargs)
        d["cases"] = [
            {
                "when": cond._compiler_dispatch(self, **kwargs),
                "then": result._compiler_dispatch(self, **kwargs),
            }
            for cond, result in clause.whens
        ]
        if clause.else_ is not None:
            d["else"] = clause.else_._compiler_dispatch(self, **kwargs)
        return d

    def visit_grouping(self, grouping, asfrom=False, **kwargs):
        return {
            "type": "grouping",
            "grouping": grouping.element._compiler_dispatch(self, **kwargs),
        }

    def visit_join(self, join, asfrom=False, **kwargs):
        if join.full:
            join_type = "FULL OUTER JOIN"
        elif join.isouter:
            join_type = "LEFT OUTER JOIN"
        else:
            join_type = "JOIN "
        return {
            "type": "join",
            "join_type": join_type,
            "left": join.left._compiler_dispatch(self, asfrom=True, **kwargs),
            "right": join.right._compiler_dispatch(self, asfrom=True, **kwargs),
            "on": join.onclause._compiler_dispatch(self, **kwargs),
        }

    def bindparam_string(self, name, **kw):
        # Bind parameters are replaced by the cursor, which calls this callable
        # with the parameter dictionary of the execution.
        return lambda params: params[name]

    def render_literal_value(self, value, type_):
        # Values stay Python objects; they are serialised as JSON, not as SQL.
        return value

    def visit_not_in_op_binary(self, binary, operator, **kw):
        # SQLAlchemy wraps NOT IN in brackets because of how it renders the
        # empty case; a document needs no bracketing.
        return self._generate_generic_binary(binary, OPERATORS[operator], **kw)

    def visit_getitem_binary(self, binary, operator, **kw):
        return {
            "type": "operator",
            "operator": "getitem",
            "operands": [
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            ],
        }

    def visit_like_op_binary(self, binary, operator, **kw):
        return {
            "type": "operator",
            "operator": "like",
            "operands": [
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            ],
        }

    def visit_slice(self, element, **kw):
        return {
            "type": "slice",
            "start": self.process(element.start, **kw),
            "stop": self.process(element.stop, **kw),
        }

    def visit_alias(
        self, alias, asfrom=False, ashint=False, iscrud=False, fromhints=None, **kwargs
    ):
        if not (asfrom or ashint):
            return alias.original._compiler_dispatch(self, **kwargs)

        if isinstance(alias.name, elements._truncated_label):
            alias_name = self._truncated_identifier("alias", alias.name)
        else:
            alias_name = alias.name

        if ashint:
            return self.preparer.format_alias(alias, alias_name)

        jsn = alias.original._compiler_dispatch(self, asfrom=True, **kwargs)
        jsn["alias"] = self.preparer.format_alias(alias, alias_name)
        return jsn

    def visit_table(
        self, table, asfrom=False, iscrud=False, ashint=False, fromhints=None, **kwargs
    ):
        if not (asfrom or ashint):
            raise exc.CompileError(f"Cannot compile table {table.name} in this context")
        return {
            "type": "table",
            "schema": getattr(table, "schema", None) or DEFAULT_SCHEMA,
            "table": table.name,
        }

    def visit_select(
        self,
        select_stmt,
        asfrom=False,
        insert_into=False,
        fromhints=None,
        compound_index=None,
        select_wraps_for=None,
        lateral=False,
        from_linter=None,
        **kwargs,
    ):
        """
        Compile a SELECT statement into an ``advanced/search`` document.

        This follows :meth:`sqlalchemy.sql.compiler.SQLCompiler.visit_select`,
        but assembles a dict instead of a string.
        """
        kwargs["within_columns_clause"] = False

        compile_state = select_stmt._compile_state_factory(select_stmt, self, **kwargs)
        kwargs["ambiguous_table_name_map"] = compile_state._ambiguous_table_name_map
        select_stmt = compile_state.statement

        toplevel = not self.stack
        if toplevel and not self.compile_state:
            self.compile_state = compile_state

        entry = self._default_stack_entry if toplevel else self.stack[-1]

        populate_result_map = need_column_expressions = (
            toplevel
            or entry.get("need_result_map_for_compound", False)
            or entry.get("need_result_map_for_nested", False)
        )
        if compound_index:
            populate_result_map = False
        if not populate_result_map and "add_to_result_map" in kwargs:
            del kwargs["add_to_result_map"]

        froms = self._setup_select_stack(
            select_stmt, compile_state, entry, asfrom, lateral, compound_index
        )

        column_clause_args = kwargs.copy()
        column_clause_args.update(
            {"within_label_clause": False, "within_columns_clause": False}
        )

        jsn = {"command": "advanced/search", "type": "select"}
        if select_stmt._distinct:
            jsn["distinct"] = True

        byfrom = None
        if select_stmt._hints:
            _, byfrom = self._setup_select_hints(select_stmt)

        inner_columns = [
            c
            for c in [
                self._label_select_column(
                    select_stmt,
                    column,
                    populate_result_map,
                    asfrom,
                    column_clause_args,
                    name=name,
                    proxy_name=proxy_name,
                    fallback_label_name=fallback_label_name,
                    column_is_repeated=repeated,
                    need_column_expressions=need_column_expressions,
                )
                for (
                    name,
                    proxy_name,
                    fallback_label_name,
                    column,
                    repeated,
                ) in compile_state.columns_plus_names
            ]
            if c is not None
        ]

        jsn = self._compose_select_body(
            jsn,
            select_stmt,
            compile_state,
            inner_columns,
            froms,
            byfrom,
            toplevel,
            kwargs,
        )

        self.stack.pop(-1)
        return jsn

    def _compose_select_body(
        self,
        jsn,
        select,
        compile_state,
        inner_columns,
        froms,
        byfrom,
        toplevel,
        kwargs,
    ):
        jsn["fields"] = inner_columns

        if froms:
            if select._hints:
                jsn["from"] = [
                    f._compiler_dispatch(self, asfrom=True, fromhints=byfrom, **kwargs)
                    for f in froms
                ]
            else:
                jsn["from"] = [
                    f._compiler_dispatch(self, asfrom=True, **kwargs) for f in froms
                ]

        if select._where_criteria:
            where = self._and_list(select._where_criteria, **kwargs)
            if where:
                jsn["where"] = where

        if select._group_by_clauses:
            group_by = [
                clause._compiler_dispatch(self, **kwargs)
                for clause in select._group_by_clauses
            ]
            if group_by:
                jsn["group_by"] = group_by

        if select._having_criteria:
            having = self._and_list(select._having_criteria, **kwargs)
            if having:
                jsn["having"] = having

        if select._order_by_clauses:
            jsn["order_by"] = [
                clause._compiler_dispatch(self, **kwargs)
                for clause in select._order_by_clauses
            ]

        if select._has_row_limiting_clause:
            if select._limit_clause is not None:
                jsn["limit"] = self.process(select._limit_clause, **kwargs)
            if select._offset_clause is not None:
                jsn["offset"] = self.process(select._offset_clause, **kwargs)

        if select._for_update_arg is not None:
            jsn["for_update"] = True

        return jsn

    def _and_list(self, criteria, **kwargs):
        """Compile a list of criteria into a single (possibly AND-ed) document."""
        compiled = [
            c
            for c in (clause._compiler_dispatch(self, **kwargs) for clause in criteria)
            if c
        ]
        if not compiled:
            return None
        if len(compiled) == 1:
            return compiled[0]
        return {
            "type": "operator",
            "operator": OPERATORS[operators.and_],
            "operands": compiled,
        }

    def visit_cast(self, cast, **kwargs):
        return {
            "type": "cast",
            "source": cast.clause._compiler_dispatch(self, **kwargs),
            "as": cast.typeclause._compiler_dispatch(self, **kwargs),
        }

    def visit_over(self, over, **kwargs):
        return {
            "type": "over",
            "function": over.func._compiler_dispatch(self, **kwargs),
            "clauses": [
                {"type": word, "clause": clause._compiler_dispatch(self, **kwargs)}
                for word, clause in (
                    ("PARTITION", over.partition_by),
                    ("ORDER", over.order_by),
                )
                if clause is not None and len(clause)
            ],
        }

    def visit_funcfilter(self, funcfilter, **kwargs):
        return {
            "type": "funcfilter",
            "function": funcfilter.func._compiler_dispatch(self, **kwargs),
            "where": funcfilter.criterion._compiler_dispatch(self, **kwargs),
        }

    def visit_extract(self, extract, **kwargs):
        return {
            "type": "extract",
            "field": self.extract_map.get(extract.field, extract.field),
            "expression": extract.expr._compiler_dispatch(self, **kwargs),
        }

    def visit_label(
        self,
        label,
        add_to_result_map=None,
        within_label_clause=False,
        within_columns_clause=False,
        render_label_as_label=None,
        **kw,
    ):
        # Labels are only rendered in the columns clause and in ORDER BY.
        render_label_with_as = within_columns_clause and not within_label_clause

        if isinstance(label.name, elements._truncated_label):
            labelname = self._truncated_identifier("colident", label.name)
        else:
            labelname = label.name

        d = {"type": "label", "label": labelname}
        if render_label_with_as:
            if add_to_result_map is not None:
                add_to_result_map(
                    labelname,
                    label.name,
                    (label, labelname) + label._alt_names,
                    label.type,
                )
            d["label"] = self.preparer.format_label(label, labelname)

        d["element"] = label.element._compiler_dispatch(
            self, within_columns_clause=False, **kw
        )
        return d

    def visit_function(self, func, add_to_result_map=None, **kwargs):
        if add_to_result_map is not None:
            add_to_result_map(func.name, func.name, (), func.type)

        disp = getattr(self, f"visit_{func.name.lower()}_func", None)
        if disp:
            return disp(func, **kwargs)
        return {
            "type": "function",
            "function": ".".join(
                list(func.packagenames) + [FUNCTIONS.get(func.__class__, func.name)]
            ),
            "operands": self.function_argspec(func, **kwargs),
        }

    def visit_column(
        self, column, add_to_result_map=None, include_table=True, **kwargs
    ):
        name = orig_name = column.name
        if name is None:
            raise exc.CompileError(
                "Cannot compile Column object until its 'name' is assigned."
            )

        is_literal = column.is_literal
        if not is_literal and isinstance(name, elements._truncated_label):
            name = self._truncated_identifier("colident", name)

        if add_to_result_map is not None:
            add_to_result_map(name, orig_name, (column, name, column.key), column.type)

        if is_literal:
            name = self.escape_literal_column(name)

        jsn = {"type": "column", "column": name, "is_literal": is_literal}

        table = column.table
        if table is None or not include_table or not table.named_with_column:
            return jsn

        if isinstance(table.name, elements._truncated_label):
            jsn["alias"] = self._truncated_identifier("alias", table.name)
        else:
            jsn["table"] = table.name
            jsn["schema"] = table.schema or DEFAULT_SCHEMA
        return jsn

    def visit_null(self, expr, **kw):
        return None

    def _generate_generic_binary(self, binary, opstring, **kw):
        return {
            "type": "operator",
            "operator": opstring,
            "operands": [
                binary.left._compiler_dispatch(self, **kw),
                binary.right._compiler_dispatch(self, **kw),
            ],
        }

    def _generate_generic_unary_operator(self, unary, opstring, **kw):
        return {
            "type": "operator",
            "operator": opstring,
            "operands": [unary.element._compiler_dispatch(self, **kw)],
        }

    def _generate_generic_unary_modifier(self, unary, opstring, **kw):
        return {
            "type": "modifier",
            "operator": opstring,
            "operands": [unary.element._compiler_dispatch(self, **kw)],
        }

    def visit_isfalse_unary_operator(self, element, operator, **kw):
        return {
            "type": "operator",
            "operator": "not",
            "operands": [self.process(element.element, **kw)],
        }

    def order_by_clause(self, select, **kw):
        return [
            clause._compiler_dispatch(self, **kw) for clause in select._order_by_clauses
        ]

    def for_update_clause(self, select, **kw):
        return {"for_update": True}


class OETypeCompiler(PGTypeCompiler):
    """Type compiler rendering FLOAT as the API documents it."""

    def visit_FLOAT(self, type_, **kw):
        if type_.asdecimal:
            d = {"type": "datatype", "datatype": "FLOAT", "kwargs": {"asdecimal": True}}
            if not type_.precision:
                d["precision"] = type_.precision
            return d
        if not type_.precision:
            return "FLOAT"
        return f"FLOAT({type_.precision})"

"""
AST Fact Extractor Module

Extracts provable facts from Java method code using javalang parser.
No inference, only structural information visible in the AST.
"""

import javalang
from typing import Dict, List, Set, Optional, Any
import re


class ASTFactExtractor:
    """Extracts structural facts from Java method code."""
    
    def __init__(self):
        # Legacy fields (maintained for backward compatibility)
        self.fields_read: Set[str] = set()
        self.fields_written: Set[str] = set()
        self.method_calls: List[str] = []
        self.null_checks: List[str] = []
        self.boundary_checks: List[str] = []
        self.exceptions_thrown: List[str] = []
        self.returns_structure: List[str] = []
        self.side_effect_evidence: List[str] = []
        
        # NEW: Richer semantic facts
        self.symbols: Dict[str, Dict[str, str]] = {}  # {name: {"kind": "param|local", "type": "..."}}
        self.variables: Dict[str, Dict[str, Any]] = {}  # Variable usage summaries
        self.scenarios: List[Dict[str, Any]] = []  # Execution scenarios
        self.exceptions: Dict[str, List[Dict[str, Any]]] = {"thrown": []}  # Structured exceptions
        self.return_summary: Dict[str, Any] = {}  # Return behavior summary
        
        # Internal tracking for AST traversal
        self.method_code: str = ""  # Store method code for position-based extraction
        self.method_node: Optional[javalang.tree.MethodDeclaration] = None
        self.current_loop_context: Optional[str] = None  # Track if we're in a loop
    
    def extract_facts(self, method_code: str) -> Dict[str, Any]:
        """
        Extract all required AST facts from method code.
        
        Args:
            method_code: Complete Java method source code
            
        Returns:
            Dictionary containing all extracted facts
        """
        try:
            # Wrap method in fake class for proper AST parsing
            # javalang.parse.parse() expects a full compilation unit, not just a method
            fake_class = f"class Dummy {{\n{method_code}\n}}"
            tree = javalang.parse.parse(fake_class)
            
            # Reset state
            self._reset()
            self.method_code = method_code  # Store for position-based extraction
            
            # Extract method signature
            method_signature = self._extract_method_signature(method_code)
            parameters = self._extract_parameters(method_code)
            return_type = self._extract_return_type(method_code)
            
            # NEW: Step 1.2 - Build symbol table and find method node
            self._build_symbol_table(tree, parameters)
            
            # NEW: Step 1.3 - Initialize variable usage summaries
            self._initialize_variable_summaries()
            
            # Walk the AST to extract facts (enhanced traversal)
            for path, node in tree:
                self._visit_node_enhanced(node, path)
            
            # NEW: Step 1.4 - Extract execution scenarios
            self._extract_scenarios(tree)
            
            # NEW: Step 1.5 - Improve exceptions modeling
            self._extract_structured_exceptions(tree)
            
            # NEW: Step 1.6 - Return behavior summary
            self._extract_return_summary(tree, return_type)
            
            # Also extract from source code directly (for patterns not easily caught by AST)
            self._extract_from_source(method_code)
            
            # Build result dictionary with backward compatibility
            result = {
                # Legacy fields (maintained for backward compatibility)
                "method_signature": method_signature,
                "parameters": parameters,
                "return_type": return_type,
                "fields_read": sorted(list(self.fields_read)),
                "fields_written": sorted(list(self.fields_written)),
                "method_calls": sorted(self.method_calls),
                "null_checks": sorted(self.null_checks),
                "boundary_checks": sorted(self.boundary_checks),
                "exceptions_thrown": sorted(self.exceptions_thrown),
                "synchronized_method": "synchronized" in method_code,
                "returns_structure": sorted(self.returns_structure),
                "side_effect_evidence": sorted(self.side_effect_evidence),
                
                # NEW: Richer semantic facts
                "symbols": self.symbols,
                "variables": self.variables,
                "scenarios": self.scenarios,
                "exceptions": self.exceptions,
                "return_summary": self.return_summary
            }
            
            return result
        except Exception as e:
            # Fallback: extract basic facts from source code
            return self._extract_basic_facts(method_code)
    
    def _reset(self):
        """Reset all extraction state."""
        # Legacy fields
        self.fields_read = set()
        self.fields_written = set()
        self.method_calls = []
        self.null_checks = []
        self.boundary_checks = []
        self.exceptions_thrown = []
        self.returns_structure = []
        self.side_effect_evidence = []
        
        # NEW: Richer semantic facts
        self.symbols = {}
        self.variables = {}
        self.scenarios = []
        self.exceptions = {"thrown": []}
        self.return_summary = {}
        
        # Internal tracking
        self.method_code = ""
        self.method_node = None
        self.current_loop_context = None
    
    def _extract_method_signature(self, method_code: str) -> str:
        """Extract method signature from source code."""
        lines = method_code.strip().split("\n")
        first_line = lines[0].strip()
        # Extract up to opening brace
        if "{" in first_line:
            return first_line.split("{")[0].strip()
        return first_line
    
    def _extract_parameters(self, method_code: str) -> List[str]:
        """Extract parameter list from method signature."""
        # Find parameters between parentheses
        match = re.search(r'\(([^)]*)\)', method_code.split("{")[0])
        if not match:
            return []
        
        params_str = match.group(1).strip()
        if not params_str:
            return []
        
        params = []
        for param in params_str.split(","):
            param = param.strip()
            if param:
                # Format: "type name" or "final type name"
                parts = param.split()
                if len(parts) >= 2:
                    param_name = parts[-1]
                    param_type = " ".join(parts[:-1])
                    params.append(f"{param_name}:{param_type}")
        
        return params
    
    def _extract_return_type(self, method_code: str) -> str:
        """Extract return type from method signature."""
        first_line = method_code.strip().split("\n")[0]
        # Remove modifiers and method name
        parts = first_line.split()
        # Find return type (usually before method name)
        for i, part in enumerate(parts):
            if "(" in part:
                # Return type is before this
                if i > 0:
                    return parts[i-1]
        return "void"
    
    def _visit_node(self, node, path):
        """Visit AST node and extract relevant facts (legacy method for backward compatibility)."""
        if isinstance(node, javalang.tree.MemberReference):
            # Field access: this.field or obj.field
            if node.member:
                if node.qualifier == "this" or not node.qualifier:
                    self.fields_read.add(node.member)
        
        elif isinstance(node, javalang.tree.Assignment):
            # Field writes
            if isinstance(node.path, javalang.tree.MemberReference):
                if node.path.member:
                    if node.path.qualifier == "this" or not node.path.qualifier:
                        self.fields_written.add(node.path.member)
        
        elif isinstance(node, javalang.tree.MethodInvocation):
            # Method calls
            method_name = node.member or ""
            if method_name:
                self.method_calls.append(method_name)
        
        elif isinstance(node, javalang.tree.ThrowStatement):
            # Exceptions thrown
            if node.expression:
                if isinstance(node.expression, javalang.tree.ClassCreator):
                    exception_type = node.expression.type.name
                    self.exceptions_thrown.append(exception_type)
        
        elif isinstance(node, javalang.tree.ReturnStatement):
            # Return structure
            if node.expression:
                if isinstance(node.expression, javalang.tree.ClassCreator):
                    self.returns_structure.append("new_object")
                elif isinstance(node.expression, javalang.tree.MethodInvocation):
                    self.returns_structure.append("method_call_result")
    
    def _build_symbol_table(self, tree, parameters: List[str]):
        """
        Step 1.2: Build symbol table from method parameters and local variables.
        
        Args:
            tree: Parsed AST tree
            parameters: List of parameter strings (format: "name:type")
        """
        # Extract parameters from parameter list
        for param_str in parameters:
            if ':' in param_str:
                param_name, param_type = param_str.split(':', 1)
                param_name = param_name.strip()
                param_type = param_type.strip()
                self.symbols[param_name] = {"kind": "param", "type": param_type}
        
        # Find method declaration and extract local variables
        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                self.method_node = node
                
                # Extract parameters from AST (more accurate than regex)
                if hasattr(node, 'parameters') and node.parameters:
                    for param in node.parameters:
                        param_name = param.name
                        # Reconstruct type string
                        if hasattr(param.type, 'name'):
                            param_type = param.type.name
                            if hasattr(param.type, 'dimensions') and param.type.dimensions:
                                param_type += "[]" * len(param.type.dimensions)
                        else:
                            param_type = str(param.type)
                        
                        self.symbols[param_name] = {"kind": "param", "type": param_type}
                
                # Extract local variables from method body
                if hasattr(node, 'body') and node.body:
                    self._extract_local_variables(node.body)
                break
    
    def _extract_local_variables(self, body):
        """Extract local variable declarations from method body."""
        if not body:
            return
        
        # Handle different body types
        statements = []
        if isinstance(body, javalang.tree.BlockStatement):
            statements = body.statements if hasattr(body, 'statements') else []
        elif isinstance(body, list):
            statements = body
        
        for stmt in statements:
            if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
                var_type = self._reconstruct_type_string(stmt.type)
                if hasattr(stmt, 'declarators') and stmt.declarators:
                    for declarator in stmt.declarators:
                        var_name = declarator.name
                        self.symbols[var_name] = {"kind": "local", "type": var_type}
    
    def _reconstruct_type_string(self, type_node) -> str:
        """Reconstruct type string from AST type node."""
        if hasattr(type_node, 'name'):
            type_str = type_node.name
            if hasattr(type_node, 'dimensions') and type_node.dimensions:
                type_str += "[]" * len(type_node.dimensions)
            if hasattr(type_node, 'arguments') and type_node.arguments:
                # Generic type arguments
                type_str += "<...>"  # Simplified for now
            return type_str
        return str(type_node)
    
    def _initialize_variable_summaries(self):
        """
        Step 1.3: Initialize variable usage summaries for all symbols.
        """
        for name, symbol_info in self.symbols.items():
            if symbol_info["kind"] in ["param", "local"]:
                self.variables[name] = {
                    "kind": symbol_info["kind"],
                    "null_checked": False,
                    "null_checked_positions": [],
                    "dereferenced": False,
                    "dereference_examples": [],
                    "assigned_to_field": False,
                    "consumed_by_iteration": False,
                    "reassigned": False,
                    "reassignment_examples": []
                }
    
    def _visit_node_enhanced(self, node, path):
        """
        Enhanced node visitor that tracks variable usage in addition to legacy facts.
        """
        # Call legacy visitor for backward compatibility
        self._visit_node(node, path)
        
        # NEW: Track variable dereferences
        if isinstance(node, javalang.tree.MethodInvocation):
            # Check if qualifier is a variable
            if hasattr(node, 'qualifier') and node.qualifier:
                if isinstance(node.qualifier, javalang.tree.Name):
                    var_name = node.qualifier.name
                    if var_name in self.variables:
                        self.variables[var_name]["dereferenced"] = True
                        method_name = node.member or ""
                        example = f"{var_name}.{method_name}()"
                        if example not in self.variables[var_name]["dereference_examples"]:
                            self.variables[var_name]["dereference_examples"].append(example)
        
        elif isinstance(node, javalang.tree.MemberReference):
            # Field access via variable
            if hasattr(node, 'qualifier') and node.qualifier:
                if isinstance(node.qualifier, javalang.tree.Name):
                    var_name = node.qualifier.name
                    if var_name in self.variables:
                        self.variables[var_name]["dereferenced"] = True
                        member = node.member or ""
                        example = f"{var_name}.{member}"
                        if example not in self.variables[var_name]["dereference_examples"]:
                            self.variables[var_name]["dereference_examples"].append(example)
        
        # NEW: Track null checks
        if isinstance(node, javalang.tree.BinaryOperation):
            if node.operator in ["==", "!="]:
                # Check if one operand is null and the other is a variable
                left = node.operandl
                right = node.operandr
                
                var_name = None
                if isinstance(left, javalang.tree.Name):
                    var_name = left.name
                elif isinstance(right, javalang.tree.Name):
                    var_name = right.name
                
                # Check if other operand is null literal
                is_null_check = (
                    (isinstance(left, javalang.tree.Literal) and left.value == "null") or
                    (isinstance(right, javalang.tree.Literal) and right.value == "null")
                )
                
                if var_name and var_name in self.variables and is_null_check:
                    self.variables[var_name]["null_checked"] = True
                    self.variables[var_name]["null_checked_positions"].append("if_condition")
        
        # NEW: Track assignments and reassignments
        if isinstance(node, javalang.tree.Assignment):
            # Check if left side is a field assignment
            if isinstance(node.path, javalang.tree.MemberReference):
                # Field assignment: this.field = ... or field = ...
                if hasattr(node.path, 'qualifier') and node.path.qualifier == "this":
                    # Check if right side contains a variable
                    right_expr = node.value
                    var_name = self._extract_variable_name_from_expression(right_expr)
                    if var_name and var_name in self.variables:
                        self.variables[var_name]["assigned_to_field"] = True
            
            # Check if left side is a variable reassignment
            if isinstance(node.path, javalang.tree.Name):
                var_name = node.path.name
                if var_name in self.variables:
                    self.variables[var_name]["reassigned"] = True
                    # Extract reassignment example from source
                    if hasattr(node, 'position') and node.position:
                        # Use position to extract snippet
                        example = self._extract_snippet_at_position(node.position)
                        if example:
                            self.variables[var_name]["reassignment_examples"].append(example)
        
        # NEW: Track iterator consumption patterns
        if isinstance(node, (javalang.tree.WhileStatement, javalang.tree.ForStatement)):
            self.current_loop_context = "loop"
            # Check condition for hasNext() pattern
            if isinstance(node, javalang.tree.WhileStatement) and hasattr(node, 'condition'):
                iterator_var = self._extract_iterator_from_condition(node.condition)
                if iterator_var and iterator_var in self.variables:
                    # Check body for next() calls
                    if hasattr(node, 'body'):
                        has_next_call = self._has_next_call_in_body(node.body, iterator_var)
                        if has_next_call:
                            self.variables[iterator_var]["consumed_by_iteration"] = True
            self.current_loop_context = None
    
    def _extract_variable_name_from_expression(self, expr) -> Optional[str]:
        """Extract variable name from an expression node."""
        if isinstance(expr, javalang.tree.Name):
            return expr.name
        elif isinstance(expr, javalang.tree.MemberReference):
            if hasattr(expr, 'qualifier') and isinstance(expr.qualifier, javalang.tree.Name):
                return expr.qualifier.name
        return None
    
    def _extract_snippet_at_position(self, position) -> str:
        """Extract code snippet at given position (simplified - uses method_code)."""
        # Simplified: just return a placeholder
        # Full implementation would use position.line and position.column
        return ""
    
    def _extract_iterator_from_condition(self, condition) -> Optional[str]:
        """Extract iterator variable name from loop condition (e.g., iterator.hasNext())."""
        if isinstance(condition, javalang.tree.MethodInvocation):
            if condition.member == "hasNext" and hasattr(condition, 'qualifier'):
                if isinstance(condition.qualifier, javalang.tree.Name):
                    return condition.qualifier.name
        return None
    
    def _has_next_call_in_body(self, body, iterator_var: str) -> bool:
        """Check if body contains iterator.next() call for the given iterator variable."""
        if not body:
            return False
        
        statements = []
        if isinstance(body, javalang.tree.BlockStatement):
            statements = body.statements if hasattr(body, 'statements') else []
        elif isinstance(body, list):
            statements = body
        
        for stmt in statements:
            if isinstance(stmt, javalang.tree.MethodInvocation):
                if stmt.member == "next" and hasattr(stmt, 'qualifier'):
                    if isinstance(stmt.qualifier, javalang.tree.Name):
                        if stmt.qualifier.name == iterator_var:
                            return True
        return False
    
    def _extract_scenarios(self, tree):
        """
        Step 1.4: Extract execution scenarios, especially early returns.
        
        Looks for if statements with early returns and extracts:
        - Condition text
        - Skipped operations
        - Involved parameters
        """
        if not self.method_node or not hasattr(self.method_node, 'body'):
            return
        
        statements = []
        if isinstance(self.method_node.body, javalang.tree.BlockStatement):
            statements = self.method_node.body.statements if hasattr(self.method_node.body, 'statements') else []
        elif isinstance(self.method_node.body, list):
            statements = self.method_node.body
        
        # Find if statements with early returns
        for i, stmt in enumerate(statements):
            if isinstance(stmt, javalang.tree.IfStatement):
                scenario = self._analyze_if_statement(stmt, statements, i)
                if scenario:
                    self.scenarios.append(scenario)
    
    def _analyze_if_statement(self, if_stmt, all_statements, if_index) -> Optional[Dict[str, Any]]:
        """Analyze an if statement to extract early return scenario."""
        # Extract condition text
        condition_text = self._extract_condition_text(if_stmt.condition)
        if not condition_text:
            return None
        
        # Check for early return in then branch
        has_early_return = False
        if hasattr(if_stmt, 'then_statement'):
            has_early_return = self._has_return_statement(if_stmt.then_statement)
        
        # Check for early return in else branch
        has_else_return = False
        if hasattr(if_stmt, 'else_statement') and if_stmt.else_statement:
            has_else_return = self._has_return_statement(if_stmt.else_statement)
        
        if not (has_early_return or has_else_return):
            return None
        
        # Extract skipped operations (method calls after this if statement)
        skipped_operations = []
        for j in range(if_index + 1, len(all_statements)):
            skipped = self._extract_method_calls_from_statement(all_statements[j])
            skipped_operations.extend(skipped)
        
        # Extract involved parameters
        involved_parameters = []
        for param_name in self.symbols:
            if self.symbols[param_name]["kind"] == "param":
                if param_name in condition_text or any(param_name in op for op in skipped_operations):
                    involved_parameters.append(param_name)
        
        return {
            "kind": "early_return",
            "condition": condition_text,
            "early_return": True,
            "skipped_operations": skipped_operations,
            "involved_parameters": involved_parameters
        }
    
    def _extract_condition_text(self, condition_node) -> str:
        """Extract condition text from AST node using source code slicing."""
        # Try to reconstruct from AST or use source code pattern matching
        # For BinaryOperation, reconstruct the condition
        if isinstance(condition_node, javalang.tree.BinaryOperation):
            left = self._expression_to_string(condition_node.operandl)
            right = self._expression_to_string(condition_node.operandr)
            op = condition_node.operator
            return f"{left} {op} {right}"
        elif isinstance(condition_node, javalang.tree.MemberReference):
            qualifier = condition_node.qualifier or ""
            member = condition_node.member or ""
            return f"{qualifier}.{member}" if qualifier else member
        elif isinstance(condition_node, javalang.tree.MethodInvocation):
            qualifier = condition_node.qualifier.name if hasattr(condition_node, 'qualifier') and isinstance(condition_node.qualifier, javalang.tree.Name) else ""
            member = condition_node.member or ""
            return f"{qualifier}.{member}()" if qualifier else f"{member}()"
        elif isinstance(condition_node, javalang.tree.Name):
            return condition_node.name
        else:
            # Fallback: use regex to find condition in source
            # Find if statements and extract conditions
            if_pattern = re.compile(r'if\s*\(([^)]+)\)', re.MULTILINE)
            matches = if_pattern.finditer(self.method_code)
            for match in matches:
                # Return first match as approximation
                return match.group(1).strip()
            return "condition"
    
    def _expression_to_string(self, expr) -> str:
        """Convert expression node to string representation."""
        if isinstance(expr, javalang.tree.Name):
            return expr.name
        elif isinstance(expr, javalang.tree.Literal):
            return str(expr.value)
        elif isinstance(expr, javalang.tree.MemberReference):
            qualifier = expr.qualifier or ""
            member = expr.member or ""
            return f"{qualifier}.{member}" if qualifier else member
        elif isinstance(expr, javalang.tree.MethodInvocation):
            qualifier = expr.qualifier.name if hasattr(expr, 'qualifier') and isinstance(expr.qualifier, javalang.tree.Name) else ""
            member = expr.member or ""
            return f"{qualifier}.{member}()" if qualifier else f"{member}()"
        else:
            return str(expr)
    
    def _has_return_statement(self, stmt) -> bool:
        """Check if statement or block contains a return statement."""
        if isinstance(stmt, javalang.tree.ReturnStatement):
            return True
        elif isinstance(stmt, javalang.tree.BlockStatement):
            if hasattr(stmt, 'statements'):
                return any(self._has_return_statement(s) for s in stmt.statements)
        return False
    
    def _extract_method_calls_from_statement(self, stmt) -> List[str]:
        """Extract method call strings from a statement."""
        calls = []
        if isinstance(stmt, javalang.tree.MethodInvocation):
            method_name = stmt.member or ""
            if method_name:
                calls.append(f"{method_name}(...)")
        elif isinstance(stmt, javalang.tree.BlockStatement):
            if hasattr(stmt, 'statements'):
                for s in stmt.statements:
                    calls.extend(self._extract_method_calls_from_statement(s))
        return calls
    
    def _extract_structured_exceptions(self, tree):
        """
        Step 1.5: Improve exceptions modeling with structured format.
        
        Extracts:
        - Explicit exceptions (from throw statements)
        - Implicit exceptions (from method calls like Integer.valueOf)
        - Caught exceptions (from try-catch blocks)
        """
        # Track try-catch blocks
        try_catch_blocks = []
        for path, node in tree:
            if isinstance(node, javalang.tree.TryStatement):
                try_catch_blocks.append(node)
        
        # Extract explicit exceptions from throw statements
        for path, node in tree:
            if isinstance(node, javalang.tree.ThrowStatement):
                if node.expression and isinstance(node.expression, javalang.tree.ClassCreator):
                    exception_type = node.expression.type.name
                    # Check if caught
                    caught = self._is_exception_caught(exception_type, try_catch_blocks)
                    self.exceptions["thrown"].append({
                        "type": exception_type,
                        "explicit": True,
                        "caught": caught
                    })
                    # Also add to legacy list
                    if exception_type not in self.exceptions_thrown:
                        self.exceptions_thrown.append(exception_type)
        
        # Extract implicit exceptions from method calls
        implicit_exception_map = {
            "valueOf": "NumberFormatException",
            "parseInt": "NumberFormatException",
            "parseLong": "NumberFormatException",
            "parseDouble": "NumberFormatException",
            "parseFloat": "NumberFormatException"
        }
        
        for method_call in self.method_calls:
            if method_call in implicit_exception_map:
                exception_type = implicit_exception_map[method_call]
                # Check if already added
                if not any(e["type"] == exception_type for e in self.exceptions["thrown"]):
                    caught = self._is_exception_caught(exception_type, try_catch_blocks)
                    self.exceptions["thrown"].append({
                        "type": exception_type,
                        "explicit": False,
                        "caught": caught
                    })
                    # Also add to legacy list so _derive_limitations_hints sees implicit exceptions
                    if exception_type not in self.exceptions_thrown:
                        self.exceptions_thrown.append(exception_type)
    
    def _is_exception_caught(self, exception_type: str, try_catch_blocks: List) -> bool:
        """Check if exception type is caught in any try-catch block."""
        for try_block in try_catch_blocks:
            if hasattr(try_block, 'catches') and try_block.catches:
                for catch_block in try_block.catches:
                    if hasattr(catch_block, 'parameter') and catch_block.parameter:
                        catch_type = catch_block.parameter.type.name
                        # Simple check: exact match or parent type
                        if catch_type == exception_type or catch_type == "Exception":
                            return True
        return False
    
    def _extract_return_summary(self, tree, return_type: str):
        """
        Step 1.6: Extract return behavior summary.
        
        Analyzes:
        - Return kinds (pure_value, collection_alias, new_object)
        - Whether method always returns
        - Field aliasing
        """
        return_statements = []
        for path, node in tree:
            if isinstance(node, javalang.tree.ReturnStatement):
                return_statements.append(node)
        
        kinds = []
        aliases_fields = []
        always_returns = len(return_statements) > 0  # Simplified: assume True if at least one return
        
        for ret_stmt in return_statements:
            if not ret_stmt.expression:
                continue
            
            expr = ret_stmt.expression
            
            # Check for new object construction
            if isinstance(expr, javalang.tree.ClassCreator):
                kinds.append("new_object")
            
            # Check for collection aliasing
            elif isinstance(expr, javalang.tree.MemberReference):
                if expr.qualifier == "this" or not expr.qualifier:
                    field_name = expr.member
                    # Check if return type is a collection
                    if any(coll in return_type for coll in ["List", "Set", "Map", "Collection"]):
                        kinds.append("collection_alias")
                        aliases_fields.append(field_name)
            
            # Check for method call result
            elif isinstance(expr, javalang.tree.MethodInvocation):
                kinds.append("method_call_result")
            
            # Default: pure value
            else:
                kinds.append("pure_value")
        
        # Remove duplicates
        kinds = list(set(kinds))
        
        self.return_summary = {
            "return_type": return_type,
            "kinds": kinds if kinds else ["pure_value"],
            "aliases_fields": aliases_fields,
            "always_returns": always_returns
        }
    
    def _extract_from_source(self, method_code: str):
        """Extract patterns directly from source code."""
        # Null checks: == null, != null, Objects.isNull, Objects.nonNull
        null_patterns = [
            r'==\s*null',
            r'!=\s*null',
            r'Objects\.isNull\(',
            r'Objects\.nonNull\(',
            r'if\s*\([^)]*null'
        ]
        for pattern in null_patterns:
            if re.search(pattern, method_code):
                self.null_checks.append("null_check_detected")
        
        # Boundary checks: <, >, <=, >=, length checks
        boundary_patterns = [
            r'<\s*\d+',
            r'>\s*\d+',
            r'<=\s*\d+',
            r'>=\s*\d+',
            r'\.length\s*[<>=]',
            r'\.size\(\)\s*[<>=]'
        ]
        for pattern in boundary_patterns:
            if re.search(pattern, method_code):
                self.boundary_checks.append("boundary_check_detected")
        
        # Side effects: assignments, method calls that might modify state
        if re.search(r'this\.\w+\s*=', method_code):
            self.side_effect_evidence.append("field_assignment")
        if re.search(r'\.put\(|\.add\(|\.remove\(|\.set\(', method_code):
            self.side_effect_evidence.append("collection_modification")
    
    def _extract_basic_facts(self, method_code: str) -> Dict[str, Any]:
        """Fallback extraction when AST parsing fails."""
        method_signature = self._extract_method_signature(method_code)
        parameters = self._extract_parameters(method_code)
        return_type = self._extract_return_type(method_code)
        
        # Basic pattern matching
        fields_read = set(re.findall(r'this\.(\w+)', method_code))
        fields_written = set(re.findall(r'this\.(\w+)\s*=', method_code))
        method_calls = re.findall(r'(\w+)\s*\(', method_code)
        
        # Build basic symbols and variables from parameters
        symbols = {}
        variables = {}
        for param_str in parameters:
            if ':' in param_str:
                param_name, param_type = param_str.split(':', 1)
                param_name = param_name.strip()
                param_type = param_type.strip()
                symbols[param_name] = {"kind": "param", "type": param_type}
                variables[param_name] = {
                    "kind": "param",
                    "null_checked": False,
                    "null_checked_positions": [],
                    "dereferenced": f"{param_name}." in method_code or f"{param_name}(" in method_code,
                    "dereference_examples": [],
                    "assigned_to_field": False,
                    "consumed_by_iteration": False,
                    "reassigned": False,
                    "reassignment_examples": []
                }
        
        return {
            # Legacy fields
            "method_signature": method_signature,
            "parameters": parameters,
            "return_type": return_type,
            "fields_read": sorted(list(fields_read)),
            "fields_written": sorted(list(fields_written)),
            "method_calls": sorted(list(set(method_calls))),
            "null_checks": ["null_check_detected"] if "null" in method_code.lower() else [],
            "boundary_checks": ["boundary_check_detected"] if re.search(r'[<>=]', method_code) else [],
            "exceptions_thrown": [],
            "synchronized_method": "synchronized" in method_code,
            "returns_structure": [],
            "side_effect_evidence": ["field_assignment"] if fields_written else [],
            
            # NEW: Richer semantic facts (basic fallback versions)
            "symbols": symbols,
            "variables": variables,
            "scenarios": [],
            "exceptions": {"thrown": []},
            "return_summary": {
                "return_type": return_type,
                "kinds": ["pure_value"],
                "aliases_fields": [],
                "always_returns": True
            }
        }


def extract_ast_facts(method_code: str) -> Dict[str, Any]:
    """
    Convenience function to extract AST facts from method code.
    
    Args:
        method_code: Java method source code
        
    Returns:
        Dictionary of extracted facts
    """
    extractor = ASTFactExtractor()
    return extractor.extract_facts(method_code)


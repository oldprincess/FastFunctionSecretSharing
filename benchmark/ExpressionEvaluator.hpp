#pragma once

#include <cctype>
#include <cmath>
#include <stack>
#include <stdexcept>
#include <string>

using namespace std;

class ExpressionEvaluator
{
public:
    double evaluate(const string& expr)
    {
        this->expr    = expr;
        pos           = 0;
        double result = parseExpression();
        if (pos < expr.size())
        {
            throw runtime_error("Unexpected character at end: '" +
                                expr.substr(pos) + "'");
        }
        return result;
    }

private:
    string expr;
    size_t pos;

    void skipWhitespace()
    {
        while (pos < expr.size() && isspace(expr[pos]))
        {
            pos++;
        }
    }

    double parseNumber()
    {
        skipWhitespace();
        size_t start = pos;
        if (expr[pos] == '-')
        {
            pos++;
        }
        while (pos < expr.size() && (isdigit(expr[pos]) || expr[pos] == '.'))
        {
            pos++;
        }
        string numStr = expr.substr(start, pos - start);
        try
        {
            return stod(numStr);
        }
        catch (...)
        {
            throw runtime_error("Invalid number: '" + numStr + "'");
        }
    }

    double parseFactor()
    {
        skipWhitespace();
        if (expr[pos] == '(')
        {
            pos++;
            double value = parseExpression();
            skipWhitespace();
            if (pos >= expr.size() || expr[pos] != ')')
            {
                throw runtime_error("Missing ')'");
            }
            pos++;
            return value;
        }
        else if (expr[pos] == '-' || expr[pos] == '+' || isdigit(expr[pos]))
        {
            return parseNumber();
        }
        else
        {
            throw runtime_error("Unexpected character: '" +
                                string(1, expr[pos]) + "'");
        }
    }

    double parseTerm()
    {
        double value = parseFactor();
        while (true)
        {
            skipWhitespace();
            char op = expr[pos];
            if (op == '*' || op == '/')
            {
                pos++;
                double right = parseFactor();
                if (op == '*')
                {
                    value *= right;
                }
                else
                {
                    if (right == 0.0)
                    {
                        throw runtime_error("Division by zero");
                    }
                    value /= right;
                }
            }
            else
            {
                break;
            }
        }
        return value;
    }

    double parseExpression()
    {
        double value = parseTerm();
        while (true)
        {
            skipWhitespace();
            char op = expr[pos];
            if (op == '+' || op == '-')
            {
                pos++;
                double right = parseTerm();
                if (op == '+')
                {
                    value += right;
                }
                else
                {
                    value -= right;
                }
            }
            else
            {
                break;
            }
        }
        return value;
    }
};
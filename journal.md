
# 13th & 14th of April

Investigate application and decide on endpoints to use in the MCP Server. Can't start building the MCP Server without this.
Understood the difference between internal APIs and external APIs (specifically due to authentication/authorization protocols).
Identfied 11 of the endpoints (external) that will be used in the MCP Server (GET, PUT, POST and PATCH operations, no Delete operations were added as endpoints).

# 15th - 18th of April

Tested the selected endpoints using Bruno so that i know the format of the requests that it requires before building in the MCP Server. 
Identified possible issues if the requests are malformed, the aunthentication/authorization method required to perform requests, the body of each Object (GlobalSettings, Scope, Channel, Article) that i'm using. 
Based on trial-and-error by using Bruno understood exactly how to build the requests, and followed application documentation and logs to determine the right course of action.
Built a log of all of the requests format and examples. 

# 19th of April

Started building the MCP Server based on all of the endpoints selected. I'm using the FastMCP Python SDK to build this MCP Server. Started by incrementally implementing the 11 tools (each tool maps to an endpoint). Tested each tool using different scenarios to understand and investigate possible errors in the tools/requests that were called based on LLM decisions (parameters that he chose to fill or didn't). Each tool was implemented with careful prompt engineering following specific rules such as: Tool descriptions lead with the trigger, never with implementations, also includes a negative case where there's a peer tool that could be confused with (e.g. Use this tool when..). Parameter / field descirptions describe constraints, format and valid input shape clearly. 
Since each request requires a Bearer Token I had to implement an HTTP Request structure that packaged the Bearer Token (this was setup as an ENV).
More complex APIs required more engineering and creative solutions:
Update tools were initially designed with the full payload of the object as a parameter that was to be updated and this would be directly sent as part of the request. The LLM internally had to call a GET operation to fectch, then merge manually before calling. This is one the most architecturally difficult tools to implement. 
Nested fields were passed as raw dictionaries with a shape hint descriptions, so the LLM can still construct them when asked.
No prompts or resources were implemented at this stage. 
In the PATCH operation the tool reads flat kwards and creates groups, only including groups that have at least one non-null field. This follows the API's PATCH semantics and lets the LLM change one particular aspect at a time without wiping the rest.
When creating tools in the MCP Server for complex enterprise APIs that require extensive request formats, most of the engineering effort is spent on trying to apply consistency, error-handling and error-prevention on the whole process (from tool description, parameter description, choice of parameter types used, how requests are compiled and sent, all while providing assurance and inambiguity to the LLM to prevent errors) by using a strong architecture due to the possibility and uncertainty that the LLM introduces when calling and filling these tool parameters. A lot of the decisions require API engineering and deep API understanding. 

# 20th of April

After the initial development of the MCP Server I decided to try to improve and update it. These issues came from pushing the complexity onto the LLM that the server should handle itself.
One of the biggest improvements / changes is the update tools. I decided to apply a patch-style. The server does the GET, the merge and the PUT itself, so the LLM only has to worry about what parameters want to change and it works.
Another architectural decisions is to use Pydantic Basemodel for the nested inputs instead of the previously used dictionaries. After some research, found out that FastMCP auto-generates proper JSON Schema from them, so the LLM sees a structured contract instead of reading a prose like "shape: {title:str, icon:str|null, links:[…]}". The improvement on tool-call accuracy on complex inputs was noticeable. 
GET tools were implemented with a specific ID as a parameter, but Users almost always use names rather than using IDs, so I decided to add two additional tools (one for each GET tool that was implemented): resolve tools - used to search and filter results by the name that was provided in order to find the correct item (candidates incase of more than one matches).
Since the endpoints required a Bearer Token for each request I decided to add another a health tool that lets the LLM verify auth at session start because these tokens expire silently.
The MCP Server currently has 14 tools and 0 prompts/resources.
In order to prevent Errors from the result of the requests to break the MCP Server I decided to return Errors as JSON envelopes and not raised exceptions. The LLM can see a parseable result and the LLM can relay it to the user instead of retrying blindly. Also decided to add a description associated with each error so the LLM can inform the user with something useful instead of a raw code (e.g. 403).
Previously for each tool call I was building a new TLS session, to avoid this i decided use a Lazily-initialized shared async HTTP client.
Additionally added server-side validation by using Annotated with string constraints to specific types.
This is one the biggest lessons while applying these improvements: trying to understand the role that the server and the LLM should have, and try to spread responsibilities accordingly. Never putting either the LLM or the server in a position where they're doing a job they're not suited for. 
Things I considered and rejected:
Delete tools - these weren't in the API spec that was available, so i'm not going to invent them.
Batch tools - creating prompts might be the right place for orchestration instead of more complex tools.


# 23rd - 26th of April

Another round of updates on the MCP Server.
After anlyzing token consumption from the logs of each LLM run, I decided to implement a technique to reduce token overhead: trimming the resulst from a request to only elements that actually are relevant and useful for the LLM as feedback on the operation. e.g.(The health tool only has to return yes or no and not the whole payload into the context). Applied this technique to all of the tools that made sense. 
Fixed a payload issue in update tools because the GET response contains fields that the PUT endpoint doesn't accept back and I was merging everything that the GET returned including server-computed fields. The fix is to explicitely allowlist only the fields the PUT endpoints accepts, stripping everything else before sending.
Built first prompt for orcherstration of workflow - Specific for the workflow that is required: to create a fully customized hub from a client's profile with a scope, customized scope settings, customized global settings, a number of channels and articles, FAQs, etc.. It's split into 5 phases: The first phase is to analyse and produce a structured workspace design proposal including the scope configuration, channel architecture, content calendar and hub-level polish. The second phase is set to build the scope. The third phase builds the channels, the fourth creates the articles and the last phase does the hub-wide polish. All of the tools that he has to call during this workflow are specified in the prompt. 

# 27th of April

Big update to the prompt. Realized that through a very complex orchestration workflow that required design + execution and multiple tool calls, a big prompt isn't going to solve my problem.
The LLM had several issues trying to keep up with 